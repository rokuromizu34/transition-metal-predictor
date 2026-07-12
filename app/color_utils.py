import math

_CMF_WL = list(range(380, 785, 5))

_CMF_X = [
    0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,
    0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,
    0.290800,0.251100,0.195360,0.142100,0.095640,0.057950,0.032010,0.014700,
    0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,
    0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,
    0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,
    0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,
    0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,
    0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,
    0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,
    0.000042,
]

_CMF_Y = [
    0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,
    0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,
    0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,
    0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,
    0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,
    0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,
    0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,
    0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,
    0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,
    0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,
    0.000015,
]

_CMF_Z = [
    0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,
    0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,
    1.669200,1.528100,1.287640,1.041900,0.812950,0.616200,0.465180,0.353300,
    0.272000,0.212300,0.158200,0.111700,0.078250,0.057250,0.042160,0.029840,
    0.020300,0.013400,0.008750,0.005750,0.003900,0.002750,0.002100,0.001800,
    0.001650,0.001400,0.001100,0.001000,0.000800,0.000600,0.000340,0.000240,
    0.000190,0.000100,0.000050,0.000030,0.000020,0.000010,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,
]


def _gaussian(wl, center, fwhm):
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return math.exp(-0.5 * ((wl - center) / sigma) ** 2)


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _linear_to_srgb(c):
    c = _clamp01(c)
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _xyz_to_hex(X, Y, Z):
    """XYZ D65 → sRGB hex."""
    r_lin =  3.2406*X - 1.5372*Y - 0.4986*Z
    g_lin = -0.9689*X + 1.8758*Y + 0.0415*Z
    b_lin =  0.0557*X - 0.2040*Y + 1.0570*Z
    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    return "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))

def _boost_saturation(hex_color: str, factor: float = 2.5) -> str:
    """Увеличивает насыщенность цвета в HSV пространстве."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
    
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    
    if delta < 1e-6:
        return hex_color  # серый — не трогаем
    
    # HSV
    sat = delta / cmax
    sat_new = min(sat * factor, 1.0)
    scale = sat_new / sat
    
    # масштабируем отклонение от максимума
    r2 = cmax - (cmax - r) * scale
    g2 = cmax - (cmax - g) * scale
    b2 = cmax - (cmax - b) * scale
    
    r2 = max(0.0, min(1.0, r2))
    g2 = max(0.0, min(1.0, g2))
    b2 = max(0.0, min(1.0, b2))
    
    return "#{:02X}{:02X}{:02X}".format(int(r2*255), int(g2*255), int(b2*255))
    # Проверенные экспериментальные цвета (hex) для известных комплексов
# Ключ: (metal, ox_state, ligand, geometry)
KNOWN_COLORS = {
    ('Co', 3, 'NH3', 'octahedral'):    ('#FEAE01', '#0050FF', 'yellow-orange'),
    ('Co', 2, 'H2O', 'octahedral'):    ('#FF6B9D', '#00C44F', 'pink'),
    ('Cu', 2, 'H2O', 'octahedral'):    ('#1E90FF', '#FF6B00', 'blue'),
    ('Cr', 3, 'H2O', 'octahedral'):    ('#7B2FBE', '#4CBB17', 'violet'),
    ('Ni', 2, 'H2O', 'octahedral'):    ('#2ECC40', '#CC2ECC', 'green'),
    ('Fe', 3, 'H2O', 'octahedral'):    ('#E8A838', '#1757C7', 'yellow-brown'),
    ('Fe', 2, 'H2O', 'octahedral'):    ('#90EE90', '#FF69B4', 'pale green'),
    ('Mn', 2, 'H2O', 'octahedral'):    ('#FFB3DE', '#004CAF', 'pale pink'),
    ('Ti', 3, 'H2O', 'octahedral'):    ('#9B59B6', '#4CBB17', 'violet'),
    ('Co', 2, 'Cl',  'tetrahedral'):   ('#1E90FF', '#FF6B00', 'blue'),
    ('Cu', 2, 'NH3', 'square_planar'): ('#1464A0', '#E8A838', 'deep blue'),
}
def _rgb_to_color_name(hex_color: str) -> str:
    """RGB hex → точное название цвета через HSV."""
    h = hex_color.lstrip('#')
    r = int(h[0:2], 16) / 255
    g = int(h[2:4], 16) / 255
    b = int(h[4:6], 16) / 255

    cmax  = max(r, g, b)
    cmin  = min(r, g, b)
    delta = cmax - cmin

    # почти белый
    if cmax > 0.90 and delta < 0.08:
        return "colorless / very pale"
    # очень тёмный
    if cmax < 0.12:
        return "black / very dark"
    # ненасыщенный
    if delta < 0.10:
        return "pale grey"

    # hue 0-360
    if delta == 0:
        hue = 0.0
    elif cmax == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        hue = 60 * (((b - r) / delta) + 2)
    else:
        hue = 60 * (((r - g) / delta) + 4)

    sat = delta / cmax  # 0-1

    # префикс насыщенности
    if sat < 0.25:
        prefix = "light "
    elif sat < 0.50:
        prefix = ""
    elif sat < 0.75:
        prefix = ""
    else:
        prefix = "deep "

    # базовый цвет по hue
    if hue < 8 or hue >= 352:
        base = "red"
    elif hue < 20:
        base = "red-orange"
    elif hue < 40:
        base = "orange"
    elif hue < 55:
        base = "yellow-orange"
    elif hue < 70:
        base = "yellow"
    elif hue < 80:
        base = "yellow-green"
    elif hue < 160:
        base = "green"
    elif hue < 185:
        base = "cyan"
    elif hue < 210:
        base = "light blue"
    elif hue < 255:
        base = "blue"
    elif hue < 275:
        base = "blue-violet"
    elif hue < 295:
        base = "violet"
    elif hue < 320:
        base = "purple"
    elif hue < 340:
        base = "pink" if sat < 0.65 else "magenta"
    else:
        base = "pink" if sat < 0.55 else "red-pink"

    return prefix + base

def spectrum_to_hex(lambda_max_nm: float, fwhm_nm: float = 120.0):
    """
    λmax → (hex_perceived, hex_absorbed, color_name)
    fwhm увеличен до 120 нм для реалистичной насыщенности.
    """
    T, A = [], []
    for wl in _CMF_WL:
        a = _gaussian(wl, lambda_max_nm, fwhm_nm)
        # ограничиваем поглощение: реальный комплекс не поглощает 100%
        a = min(a, 0.95)
        A.append(a)
        T.append(1.0 - a)

    # XYZ пропускания (воспринимаемый цвет)
    Xt = sum(t * x for t, x in zip(T, _CMF_X))
    Yt = sum(t * y for t, y in zip(T, _CMF_Y))
    Zt = sum(t * z for t, z in zip(T, _CMF_Z))

    # XYZ поглощения
    Xa = sum(a * x for a, x in zip(A, _CMF_X))
    Ya = sum(a * y for a, y in zip(A, _CMF_Y))
    Za = sum(a * z for a, z in zip(A, _CMF_Z))

    # нормировка по максимальному каналу → насыщенный цвет
    def norm_and_hex(X, Y, Z):
        m = max(X, Y, Z, 1e-6)
        return _xyz_to_hex(X / m, Y / m, Z / m)

    hex_perceived = _boost_saturation(norm_and_hex(Xt, Yt, Zt), factor=2.5)
    hex_absorbed  = _boost_saturation(norm_and_hex(Xa, Ya, Za), factor=2.5)
    
    color_name = _rgb_to_color_name(hex_perceived)
    return hex_perceived, hex_absorbed, color_name

def wavelength_to_absorbed_name(wl_nm: float) -> str:
    wl = float(wl_nm)
    if wl < 380: return "UV"
    if wl < 450: return "violet"
    if wl < 495: return "blue"
    if wl < 570: return "green"
    if wl < 590: return "yellow"
    if wl < 620: return "orange"
    if wl <= 750: return "red"
    return "IR"