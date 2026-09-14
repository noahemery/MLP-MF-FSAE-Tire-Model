"""Pacejka tire model for the FK lap sim. Standalone, numpy only.

Supersedes tire_model.py, whose hardcoded parameters are the stale
20-parameter lateral form and do not match the current fits.

    import tire_predict as tp

    tp.lateral_force(F_z=800, slip_angle_deg=5, camber_deg=0,
                     tire="16.0X7.5-10 R20 7.0")

Parameters come from Calspan TTC Round 9 rig data, fitted through magic.py
(William Young's physics) and embedded here as literals, so this module needs
nothing but numpy.

HOW ACCURATE IS IT
------------------
RMSE as a percentage of 95th-percentile force, on the data each tire was
fitted to:

    lateral        3.2 - 4.7%     usable
    longitudinal   6.1 - 7.9%     usable
    G_x combined   8.2 - 8.8%     usable
    G_y combined  10.0 - 18.3%    weakest, see below

M_x (overturning_moment) is reported differently, because it is cheap enough
to score properly: HELD-OUT error, GroupKFold by segment, as a percentage of
95th-percentile |M_x|. It averages 10.1% across the six tires, or 8.2% if
pressure_kpa is supplied. All six tires are covered, including both
16.0X7.5-10 specs, which have no combined-slip data.

There is no M_z yet, and no M_y at all -- the TTC files carry no MY channel.

GY_180X60_R20_60 is the worst fit at 18.3%. Its straight-line runs only swept
slip angle from -6 to -3 degrees, so the G_y parameters there are poorly
constrained. Calling combined_force on that tire emits a warning.

UNTESTED TIRES
--------------
get_params() raises for a tire that was not measured. A pooled fallback exists
-- one parameter set fitted across all tires -- and you can opt in with
allow_fallback=True, but it is NOT validated for a size outside the dataset.

Leave-one-tire-out testing showed geometry-based prediction does not
generalise from this dataset: on lateral and G_y a model that ignores geometry
entirely beat one that uses it. With 6 tires across 3 sizes there is not
enough spread to learn a size trend. Treat the fallback as "best available
guess", not a prediction.

SIGN CONVENTIONS
----------------
F_z is positive in compression (pass the load as a positive number).
slip_angle_deg and camber_deg are in degrees; slip_ratio is dimensionless.
Returned forces follow magic.py: F_y positive for positive slip angle,
F_x positive under drive.
"""

import warnings

import numpy as np

__all__ = ["get_params", "lateral_force", "longitudinal_force",
           "combined_force", "overturning_moment", "available_tires"]

# Number of fitted parameters per family. Stored vectors carry trailing
# metadata after this: [params..., diameter_in, F_z0] for the pure-slip
# families, bare params for the G families.
N_FITTED = {"lateral": 22, "longitudinal": 14, "gx": 7, "gy": 15,
            "mx": 3, "mxp": 6}

# Fits with a known weakness, warned about at call time.
LOW_CONFIDENCE = {
    "GY_180X60_R20_60": ("G_y for this tire is fitted from straight runs that "
                         "only span -6 to -3 degrees of slip angle; the "
                         "parameters are poorly constrained (18.3% error)"),
}

PARAM_NAMES = {
    "lateral": ["PDY1", "PDY2", "PDY3", "PCY1", "PKY1", "PKY2", "PKY3", "PKY4",
                "PKY5", "PKY6", "PKY7", "PHY1", "PHY2", "PEY1", "PEY2", "PEY3",
                "PEY4", "PEY5", "PVY1", "PVY2", "PVY3", "PVY4",
                "diameter_in", "F_z0"],
    "longitudinal": ["PDX1", "PDX2", "PCX1", "PKX1", "PKX2", "PKX3", "PHX1",
                     "PHX2", "PEX1", "PEX2", "PEX3", "PEX4", "PVX1", "PVX2",
                     "diameter_in", "F_z0"],
    "gx": ["RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1"],
    "gy": ["RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2", "RHY1",
           "RHY2", "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6"],
    "mx": ["QSX1", "QSX2", "QSX3", "diameter_in", "F_z0"],
    "mxp": ["QSX1", "QSX2", "QSX3", "QSX1p", "QSX2p", "QSX3p",
            "diameter_in", "F_z0", "P_nom"],
}

PER_TIRE = {
    "GX_180X60_R20_60": [
        10.16447330214713, -8.1414211885595937e-07, -193.57195968745933, 1.2932284921229056,
        0.73761913593036954, 0.10161120143280225, 0.0043129635943499559],
    "GX_180X60_R20_70": [
        9.4331476020495248, -5.4780352507823815e-07, 55.2098134565028, 1.2464739710887103,
        0.73280173312233921, 0.094505675728363575, 0.004618954365579464],
    "GX_205X70_R20_70": [
        9.5004023633504957, 0.74445375057983076, 86.122526760761474, 1.193197015560242,
        0.75083241784527399, 0.031906857595954609, 0.0039729331387800479],
    "GX_205X70_R20_80": [
        9.684843573405665, 0.77860956008189264, -16.164939105704107, 1.2240682625664865,
        0.7421159417217511, 0.062883264253969118, 0.0047365829514822355],
    "GY_180X60_R20_60": [
        10.071391438898495, -0.010918426861747529, -0.16796596403284234, -21.433718516156627,
        1.5383714266326927, 0.75677812839577741, -0.01656679496929693, 0.0001271645598533916,
        0.00032329565437015715, -0.012840732100954882, 0.012397035512568785, 0.15396132762524772,
        34.484441255849745, 2.7469346936156214, 0.75762190567897159],
    "GY_180X60_R20_70": [
        10.033899986391546, 4.5777326112373231, -0.081943874317848414, 6.6507740793260393,
        1.3944357483779735, 0.75336358120779456, -0.0044083235137960348, -1.7787613685920753e-05,
        0.00032668696703580004, -0.047855978768091459, 0.084759817803880574, 0.24104014236621085,
        117.81937859914947, 6.4098940607441524, 0.14582697177067042],
    "GY_205X70_R20_70": [
        13.692645621021008, 0.41456842309211139, -2.2806278219895399, 290.99031631349038,
        1.3794785303584276, 0.74945002521891191, -0.013559442328066317, -0.00017952693204616735,
        0.00097067937075658917, -0.0055720337092053322, 0.011589075994214418, -0.085649958476851459,
        48.490662394022969, 5.8297700045778784, 1.507929628007284],
    "GY_205X70_R20_80": [
        40.082358067340706, 0.056874792388813475, 67.875937093427723, 973.00802438120184,
        1.3776115282380306, 0.75562244945135393, -0.024121037947878222, -0.00087607940977159685,
        0.0024878137001858921, 0.012026198444186316, -0.012715754109317802, -1.0659988825564188,
        433.05564202874575, 2.3402330077754958, -410.50100637607238],
    "lat_160X75_R20_70": [
        2.4902287349604419, -0.21093080915194407, 9.9964517678089333, 1.5457002610420578,
        43.861959364503925, -0.80123916881810253, 48.525201585536081, -1.3315820215799663,
        11.049811948053344, -3.2343188841375978, -1.1643933583576094, 0.0024477249541572147,
        0.00033818254078767313, 0.40398643941629281, -0.137014942595998, -0.0042114025540726709,
        0.087446927512168302, -42.309582466595273, 0.026434785376001319, 0.012558265590426781,
        -1.398669688112125, 1.4233958023058939, 16, 740.59459957279432],
    "lat_160X75_R20_80": [
        2.4756151411535248, -0.2106634912955157, 14.558596591421059, 1.5311600182458067,
        49.824745435533565, -0.70708072495072027, 45.688148147677595, -1.0952326115954649,
        -8.0064021342022489, -3.3065985100659208, -1.3692822779932423, 0.0023907726124606887,
        0.0010812744044547563, 0.41179288563659305, -0.14479545535317823, -0.0021911967672839454,
        0.049281324687609068, -5.7239501209765953, 0.040477917750971031, 0.020449421294840401,
        -0.76803251211344037, 0.82019395634883541, 16, 733.28247241067879],
    "lat_180X60_R20_60": [
        2.6367345490194753, -0.3520774473242117, 5.6594209863996587, 1.3451580125411478,
        527.66324612062931, -0.42828163244827078, 36.529046599565604, -0.058615788660206683,
        17.120952693429295, -3.5403505316684276, -0.84521874542447528, 0.0027196095194624566,
        0.0016687964698660423, 0.37377641030457293, -0.03986740589142794, -0.001606916908819361,
        0.028285981891098721, -12.700367969578158, 0.035511525423892264, -0.037018613996967847,
        -0.7390438362116486, 0.89988169723685163, 18, 729.0681149040704],
    "lat_180X60_R20_70": [
        2.6203948406668509, -0.24544133163346105, 12.702495197461499, 1.4088016041509708,
        4773.0275390072284, 0.50971480922446355, 59.564930479736773, 0.007623841862803106,
        -26.359519299608884, -3.2433817521143262, -1.0830587580429931, 0.0011464574699712234,
        0.0012751604344376517, 0.39133060468774367, -0.095842454392796705, -0.003115234944553034,
        0.055404101216688498, -8.2072220373836533, 0.066442538482443442, -0.030519626815453779,
        -0.46893518530854977, 0.7667554330644416, 18, 731.34525972664892],
    "lat_205X70_R20_70": [
        2.4549285459554921, -0.28568515456789784, 11.206110639846335, 1.5713767608190996,
        1339.5874959430575, -0.83536969942712347, 54.380563782140413, -0.045464600467155669,
        35.127740099608204, -4.3451315672327588, -0.8829496035230715, 0.001646980149653824,
        -0.00019828701113119572, 0.4904143255582139, -0.0064442173505200309, -0.00028477979966442163,
        0.012015023135357912, -43.356304618595878, 0.040988905215667101, 0.0182723967878432,
        0.25209303640007774, -0.14362234763540468, 20.5, 875.32594954983892],
    "lat_205X70_R20_80": [
        2.419060723677112, -0.24527037474756366, 9.3668198781064085, 1.5781884612709696,
        1464.2653653602488, 0.92854087223166715, 28.688371054684069, 0.045351247885367603,
        -0.8106030931447602, -3.9958627484195492, -1.597959180909313, 0.0018858639174870824,
        -2.9564052592886979e-05, 0.43031246713342325, 0.063318520535360426, 0.0020412586953275861,
        -0.011793928952289361, -39.88490016077558, 0.036067439781943385, 0.035681460176964663,
        0.11352838751066623, -0.14274770361476233, 20.5, 875.60982454112479],
    "long_180X60_R20_60": [
        2.3876537665175719, -0.17792345384139421, 1.679693386141516, 48.936880601392104,
        23.011031024882403, -0.82192846581112522, -0.00026982252950198221, -0.001195909592001167,
        0.44333532676337978, -0.071663855265148013, 0.0287862503945755, 0.0099608704943689088,
        0.033044539052821967, -0.033723520528632339, 18, 707.13995369821873],
    "long_180X60_R20_70": [
        2.2804356678389417, -0.15535334031433295, 1.6476109981716642, 49.952495422569243,
        25.026372162556132, -0.89980245458920005, -0.00054257365105479783, -0.00037024931453470699,
        0.34374842644631248, -0.22520420975675778, 0.27091408795748628, -0.003367950248195964,
        0.042192032122696829, -0.04018350807903906, 18, 700.51129712805778],
    "long_205X70_R20_70": [
        2.2320860407237855, -0.054175935523180639, 1.7792298657582224, 52.475479000566551,
        -0.0077706300832156069, -0.20326043397377722, -0.0011224758274741527, -0.0015725717238564674,
        0.48729581319322551, -0.40786385711949952, -0.18893690068643623, -0.0056205128391672932,
        0.06358976278182138, 0.0091546618838674893, 20.5, 814.79210618483103],
    "long_205X70_R20_80": [
        2.211375856641927, -0.16769102395004706, 1.7635123930720176, 49.557794390121387,
        -0.00076937590111244268, -0.26404546007064433, -0.0010664666598914328, -0.0010117365962322897,
        0.50598408578129839, -0.30822402281148781, -0.093029608109662085, -0.0041430959092638294,
        0.071278296010994474, -0.0050501748300704275, 20.5, 816.5304698104751],
    "MX_160X75_R20_70": [
        -0.02770012050269322, 1.652490324067806, 0.05584257152735489, 16.0,
        740.5945995727943],
    "MX_160X75_R20_80": [
        -0.022115523736468527, 1.8817858591635122, 0.05316151119711262, 16.0,
        733.2824724106788],
    "MX_205X70_R20_70": [
        -0.024095702807115208, 1.0679677093612496, 0.05433130248089002, 20.5,
        875.3259495498389],
    "MX_205X70_R20_80": [
        -0.017846300766866628, 1.1136459328445452, 0.051717757477973025,
        20.5, 875.6098245411248],
    "MX_180X60_R20_60": [
        -0.020589715739214743, 1.1749100234015515, 0.06351418673556386, 18.0,
        729.0681149040704],
    "MX_180X60_R20_70": [
        -0.023317475882588725, 1.3998084501714587, 0.06354647084757328, 18.0,
        731.3452597266489],
    "MXP_160X75_R20_70": [
        -0.027554042861839607, 1.7077814708239276, 0.053423309476122616,
        0.006594072663491879, 0.8811975836148145, -0.04673988022237524, 16.0,
        740.5945995727943, 82.7],
    "MXP_160X75_R20_80": [
        -0.022093485679940628, 1.9325244178591734, 0.05087592062449334,
        -0.002437371199367496, 0.6795185596857604, -0.051908618748560446,
        16.0, 733.2824724106788, 82.7],
    "MXP_205X70_R20_70": [
        -0.02428884472732418, 1.090288128972769, 0.05086537911816846,
        -0.0020840231171920243, 0.32433008106628414, -0.04738844860017341,
        20.5, 875.3259495498389, 82.7],
    "MXP_205X70_R20_80": [
        -0.0177769099956056, 1.1506968896528595, 0.048583536932951366,
        -0.00031025911406085127, 0.5361847845335714, -0.0518950009820459,
        20.5, 875.6098245411248, 82.7],
    "MXP_180X60_R20_60": [
        -0.02050067896107664, 1.2045257238142122, 0.06301666373242629,
        0.04980006151633568, 1.2222438494211487, -0.034346040610581725, 18.0,
        729.0681149040704, 82.7],
    "MXP_180X60_R20_70": [
        -0.022443911332154984, 1.450339343477759, 0.06196068757721536,
        0.025062363846840387, 0.9292061657052324, -0.04136573915331362, 18.0,
        731.3452597266489, 82.7],
}

POOLED = {
    "gx": [
        9.300279095833627, 1.410088577158582, 185.08015218063721, 1.1614983967204178,
        0.80987812341637788, 0.12692672011268219, 0.0040570871674324941],
    "gy": [
        18.462913293604405, 1.7964178351998064, 0.65096040028174507, -106.2772258689788,
        1.2782647033557297, 0.81661935513315786, -0.0065344438744208196, -0.00016662793429973001,
        0.0013956368380706984, -0.00058699312871794349, 0.049411360449646918, 0.28988198757322647,
        15.123811134878338, 2.8880791826464574, -183.5497717819473],
    "lateral": [
        2.440568619369567, -0.23583177844891531, 10.342068193402566, 1.5214590272425839,
        536.35853018336968, -0.67964140039504961, 36.12302758924001, -0.089056986512374148,
        21.301756798953374, -3.8382205154530267, -1.5010653718414184, 0.0021275482292487571,
        1.0433063215794808e-07, 0.39709305499767522, -0.01224140452038984, 0.0011821196634528631,
        0.078526686119886807, -39.267943140321023, 0.027137517679999362, -0.0046260901088093446,
        0.035286214421547557, 0.94818003385582461],
    "longitudinal": [
        2.2801867824797957, -0.1438256030592773, 1.6340389730018046, 45.395593946285651,
        12.066991260934529, -0.33284637426966074, -0.0011432866227540641, -0.00016842867093082574,
        0.54956364204339425, -0.62377466810403059, 0.20171322485397314, -0.0027634238477388866,
        0.023774178722266011, 0.0074797667652471839],
    # MX pooled sets are a single fit over all six tires' data stacked
    # together, not an average of the six parameter vectors. diameter_in and
    # F_z0 here are the means across tires and are only sensible as a rough
    # stand-in for an unmeasured tire.
    "mx": [
        -0.02256515575588318, 1.2883909520435888, 0.05598792071864844,
        18.166666666666668, 780.8710367841927],
    "mxp": [
        -0.02207439014630916, 1.3343150872503409, 0.05361239365757853,
        0.01008741972902183, 0.7030905783201131, -0.044267543721476094,
        18.166666666666668, 780.8710367841927, 82.7],
}

TIRE_INDEX = {
    "16.0X7.5-10 R20 7.0": "160X75_R20_70",
    "16.0X7.5-10 R20 8.0": "160X75_R20_80",
    "18.0X6.0-10 R20 6.0": "180X60_R20_60",
    "18.0X6.0-10 R20 7.0": "180X60_R20_70",
    "20.5X7.0-13 R20 7.0": "205X70_R20_70",
    "20.5X7.0-13 R20 8.0": "205X70_R20_80",
}


# ---------------------------------------------------------------------------
# Parameter access
# ---------------------------------------------------------------------------

def available_tires():
    """Tire names this module has measured parameters for."""
    return sorted(TIRE_INDEX)


def _resolve(tire, family, allow_fallback):
    """Return (parameter vector, spec_id or None) for one tire and family."""
    if tire in TIRE_INDEX:
        prefix = {"lateral": "lat_", "longitudinal": "long_",
                  "gx": "GX_", "gy": "GY_",
                  "mx": "MX_", "mxp": "MXP_"}[family]
        spec_id = prefix + TIRE_INDEX[tire]
        if spec_id in PER_TIRE:
            if spec_id in LOW_CONFIDENCE:
                warnings.warn(LOW_CONFIDENCE[spec_id], stacklevel=3)
            return np.asarray(PER_TIRE[spec_id], dtype=float), spec_id
        # Known tire, but this family was never fitted for it -- the two
        # 16.0X7.5-10 specs have cornering data only.
        if not allow_fallback:
            raise KeyError(
                "tire %r has no %s fit (it has no straight-line data). "
                "Pass allow_fallback=True to use the pooled parameters, "
                "which are not validated for this tire." % (tire, family))
    elif not allow_fallback:
        raise KeyError(
            "unknown tire %r. Measured tires: %s. Pass allow_fallback=True "
            "to use pooled parameters, but note they are NOT validated for "
            "an untested size -- see the module docstring."
            % (tire, ", ".join(available_tires())))

    warnings.warn(
        "using pooled fallback parameters for %r (%s). These are not "
        "validated for this tire; treat the result as an estimate."
        % (tire, family), stacklevel=3)
    return np.asarray(POOLED[family], dtype=float), None


def get_params(tire, family="lateral", allow_fallback=False):
    """Fitted parameters for one tire and family, as a dict of name -> value."""
    v, _ = _resolve(tire, family, allow_fallback)
    return {n: float(x) for n, x in zip(PARAM_NAMES[family], v)}


def _core(vector, family):
    """Split a stored vector into (fitted params, F_z0)."""
    n = N_FITTED[family]
    v = np.asarray(vector, dtype=float).ravel()
    if family in ("gx", "gy"):
        return v[:n], None
    if family == "mxp":
        return v[:n], float(v[-2])      # trailing element is P_nom
    return v[:n], float(v[-1])


# ---------------------------------------------------------------------------
# The Magic Formula. Transcribed from magic.py; verify_against_magic() is the
# gate that proves these match.
# ---------------------------------------------------------------------------

def _tm_lat(F_z, alpha, gamma, p, F_z0):
    (PDY1, PDY2, PDY3, PCY1, PKY1, PKY2, PKY3, PKY4, PKY5, PKY6, PKY7,
     PHY1, PHY2, PEY1, PEY2, PEY3, PEY4, PEY5, PVY1, PVY2, PVY3, PVY4) = p
    df_z = (F_z - F_z0) / F_z0
    mu_y = (PDY1 + PDY2 * df_z) / (1 + PDY3 * gamma ** 2)
    BCD_y = (PKY1 * F_z0
             * np.sin(PKY4 * np.arctan(
                 F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0)))
             / (1 + PKY3 * gamma ** 2))
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
    K_y_gamma_0 = F_z * (PKY6 + PKY7 * df_z)
    D_y = mu_y * F_z
    B_y = BCD_y / (mu_y * F_z * PCY1 + 1e-8)
    S_hy = ((PHY1 + PHY2 * df_z)
            + (K_y_gamma_0 * gamma - S_vy_gamma) / (BCD_y + 1e-8))
    E_y = ((PEY1 + PEY2 * df_z)
           * (1 + PEY5 * gamma ** 2
              - (PEY3 + PEY4 * gamma) * np.sign(alpha + S_hy)))
    S_vy = (PVY1 + PVY2 * df_z) * F_z + S_vy_gamma
    u = B_y * (alpha + S_hy)
    Y = D_y * np.sin(PCY1 * np.arctan(u - E_y * (u - np.arctan(u)))) + S_vy
    return Y, BCD_y, mu_y


def _tm_long(F_z, s, p, F_z0):
    (PDX1, PDX2, PCX1, PKX1, PKX2, PKX3, PHX1, PHX2,
     PEX1, PEX2, PEX3, PEX4, PVX1, PVX2) = p
    df_z = (F_z - F_z0) / F_z0
    mu_x = PDX1 + PDX2 * df_z
    BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
    D_x = mu_x * F_z
    B_x = BCD_x / (mu_x * F_z * PCX1)
    S_hx = PHX1 + PHX2 * df_z
    E_x = ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2)
           * (1 - PEX4 * np.sign(s + S_hx)))
    S_vx = F_z * (PVX1 + PVX2 * df_z)
    u = B_x * (s + S_hx)
    Y = D_x * np.sin(PCX1 * np.arctan(u - E_x * (u - np.arctan(u)))) + S_vx
    return Y, BCD_x


def _gx(F_z, F_z0, s, alpha, gamma, p):
    RBX1, RBX2, RBX3, RCX1, REX1, REX2, RHX1 = p
    df_z = (F_z - F_z0) / F_z0
    B = (RBX1 + RBX3 * gamma ** 2) * np.cos(np.arctan(RBX2 * s))
    E = REX1 + REX2 * df_z
    u0 = B * RHX1
    G0 = np.cos(RCX1 * np.arctan(u0 - E * (u0 - np.arctan(u0))))
    u = B * (alpha + RHX1)
    return np.cos(RCX1 * np.arctan(u - E * (u - np.arctan(u)))) / G0


def _gy(F_z, F_z0, s, alpha, gamma, p, mu_y):
    (RBY1, RBY2, RBY3, RBY4, RCY1, REY1, REY2, RHY1, RHY2,
     RVY1, RVY2, RVY3, RVY4, RVY5, RVY6) = p
    df_z = (F_z - F_z0) / F_z0
    B = (RBY1 + RBY4 * gamma ** 2) * np.cos(np.arctan(RBY2 * (alpha - RBY3)))
    E = REY1 + REY2 * df_z
    S_h = RHY1 + RHY2 * df_z
    D_vgy = (mu_y * F_z * (RVY1 + RVY2 * df_z + RVY3 * gamma)
             * np.cos(np.arctan(RVY4 * alpha)))
    S_vgy = D_vgy * np.sin(RVY5 * np.arctan(RVY6 * s))
    u0 = B * S_h
    G0 = np.cos(RCY1 * np.arctan(u0 - E * (u0 - np.arctan(u0))))
    u = B * (s + S_h)
    return np.cos(RCY1 * np.arctan(u - E * (u - np.arctan(u)))) / G0, S_vgy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lateral_force(F_z, slip_angle_deg, camber_deg=0.0, tire=None,
                  allow_fallback=False):
    """Lateral force F_y in newtons. Pure slip (no slip ratio)."""
    v, _ = _resolve(tire, "lateral", allow_fallback)
    p, F_z0 = _core(v, "lateral")
    alpha = np.tan(np.asarray(slip_angle_deg, dtype=float) * np.pi / 180.0)
    gamma = np.sin(np.asarray(camber_deg, dtype=float) * np.pi / 180.0)
    return _tm_lat(np.asarray(F_z, dtype=float), alpha, gamma, p, F_z0)[0]


def longitudinal_force(F_z, slip_ratio, tire=None, allow_fallback=False):
    """Longitudinal force F_x in newtons. Pure slip (no slip angle)."""
    v, _ = _resolve(tire, "longitudinal", allow_fallback)
    p, F_z0 = _core(v, "longitudinal")
    return _tm_long(np.asarray(F_z, dtype=float),
                    np.asarray(slip_ratio, dtype=float), p, F_z0)[0]


def combined_force(F_z, slip_angle_deg, slip_ratio, camber_deg=0.0, tire=None,
                   allow_fallback=False):
    """Combined-slip forces. Returns (F_x, F_y) in newtons."""
    F_z = np.asarray(F_z, dtype=float)
    alpha = np.tan(np.asarray(slip_angle_deg, dtype=float) * np.pi / 180.0)
    gamma = np.sin(np.asarray(camber_deg, dtype=float) * np.pi / 180.0)
    s = np.asarray(slip_ratio, dtype=float)

    lat, _ = _resolve(tire, "lateral", allow_fallback)
    lng, _ = _resolve(tire, "longitudinal", allow_fallback)
    gxp, _ = _resolve(tire, "gx", allow_fallback)
    gyp, _ = _resolve(tire, "gy", allow_fallback)
    pl, F_z0_y = _core(lat, "lateral")
    px, F_z0_x = _core(lng, "longitudinal")

    F_y0, _, mu_y = _tm_lat(F_z, alpha, gamma, pl, F_z0_y)
    F_x0, _ = _tm_long(F_z, s, px, F_z0_x)
    G_x = _gx(F_z, F_z0_x, s, alpha, gamma, _core(gxp, "gx")[0])
    G_y, S_vgy = _gy(F_z, F_z0_y, s, alpha, gamma, _core(gyp, "gy")[0], mu_y)
    return F_x0 * G_x, F_y0 * G_y + S_vgy


def overturning_moment(F_z, slip_angle_deg, camber_deg=0.0, tire=None,
                       pressure_kpa=None, allow_fallback=False):
    """Overturning moment M_x in newton-metres.

    Fitted on pure-slip cornering data, so this is the pure-slip value: it
    takes no slip ratio. Held-out error, GroupKFold by segment, averages 10.1%
    of the 95th-percentile |M_x| across tires, or 8.2% with pressure_kpa given.

    pressure_kpa is optional. Omit it and you get the model exactly as
    magic.py's fit_MX defines it, valid around the ~83 kPa (12 psi) modal test
    pressure. Pass it and the pressure-extended parameter set is used instead,
    which tracks a real and consistent effect: the M_x lateral-force
    coefficient falls about 35% from 8 to 14 psi in every tire measured.

    The pressure-extended form is an EXTENSION, not magic.py's model, and is
    awaiting the physics owner's review. The default path is unaffected by it.
    """
    family = "mx" if pressure_kpa is None else "mxp"
    v, _ = _resolve(tire, family, allow_fallback)
    p, F_z0 = _core(v, family)
    R_0 = float(v[-3] if family == "mxp" else v[-2]) * 0.5 * 0.0254   # in -> m

    F_z = np.asarray(F_z, dtype=float)
    gamma = np.sin(np.asarray(camber_deg, dtype=float) * np.pi / 180.0)
    F_y = lateral_force(F_z, slip_angle_deg, camber_deg, tire, allow_fallback)

    if family == "mx":
        QSX1, QSX2, QSX3 = p
    else:
        P_nom = float(v[-1])
        dpi = (np.asarray(pressure_kpa, dtype=float) - P_nom) / P_nom
        QSX1 = p[0] + p[3] * dpi
        QSX2 = p[1] + p[4] * dpi
        QSX3 = p[2] + p[5] * dpi

    return F_z * R_0 * (QSX1 - QSX2 * gamma + QSX3 * F_y / F_z0)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def verify_against_magic(n=3000, seed=0, tol=1e-10, verbose=True):
    """Prove these numpy functions reproduce magic.py. Returns worst rel diff."""
    import magic
    rng = np.random.default_rng(seed)
    F_z = rng.uniform(200.0, 2000.0, n)
    alpha = np.tan(rng.uniform(-15.0, 15.0, n) * np.pi / 180.0)
    gamma = np.sin(rng.uniform(0.0, 4.0, n) * np.pi / 180.0)
    s = rng.uniform(-0.2, 0.2, n)

    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tire in available_tires():
            for fam in ("lateral", "longitudinal"):
                try:
                    v, _ = _resolve(tire, fam, False)
                except KeyError:
                    continue
                p, F_z0 = _core(v, fam)
                full = np.concatenate([p, [F_z0]])
                if fam == "lateral":
                    mine = _tm_lat(F_z, alpha, gamma, p, F_z0)[0]
                    theirs = magic.tm_lat(F_z, alpha, gamma, 1, full)[0]
                else:
                    mine = _tm_long(F_z, s, p, F_z0)[0]
                    theirs = magic.tm_long(F_z, s, 1, full)[0]
                rel = float(np.max(np.abs(mine - theirs)
                                   / np.maximum(np.abs(theirs), 1e-9)))
                worst = max(worst, rel)
                if verbose:
                    print("  %-24s %-13s max_rel=%.3e" % (tire, fam, rel))
    if verbose:
        print("\nworst relative difference: %.3e  %s"
              % (worst, "PASS" if worst < tol else "FAIL"))
    return worst


if __name__ == "__main__":
    import sys
    print("verifying tire_predict against magic.py\n")
    sys.exit(0 if verify_against_magic() < 1e-10 else 1)
