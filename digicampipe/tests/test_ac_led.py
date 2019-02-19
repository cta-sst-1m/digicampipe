import numpy as np
import tempfile

from digicampipe.instrument.light_source import ACLED

AC_LEVELS = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     25, 30, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115,
     120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190,
     195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265,
     270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340,
     345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415,
     420, 425, 430, 435, 440, 445])

N_PE = np.array(
    [[np.nan, 0.5914502096265155],
     [0.3215599666348271, 0.6192213504740821],
     [0.33428735965970013, 0.6412313426838381],
     [0.3427694029568884, 0.6394225603430103],
     [0.35900027479246355, 0.6690911659762848],
     [0.35366529637047667, 0.6901943349802222],
     [0.37777030389804456, 0.6860568691634314],
     [0.38638174809060755, 0.7101073121858349],
     [0.40150646628202896, 0.7363064306957621],
     [0.39141746474390704, 0.745809592417428],
     [0.4035977146073631, 0.7521135111900508],
     [0.42825055452306465, 0.7864085817237958],
     [0.4388369783730469, 0.8141059125751826],
     [0.44666637289842914, 0.8235057784805819],
     [0.4524745515227204, 0.8512028142659256],
     [0.4722360050259609, 0.8781308385208265],
     [0.4780836635164927, 0.8843396051992218],
     [0.48689123012699276, 0.9046981101042958],
     [0.4899174021190744, 0.9404168190851538],
     [0.512554977914036, 0.9464264334003589],
     [0.536662474518065, 0.9819403297012916],
     [0.6015249525329678, 1.1202098077589087],
     [0.69274698858735, 1.2733773331387108],
     [0.7852848818951907, 1.430143895559617],
     [1.0106427935667508, 1.7948752404980146],
     [1.1341033060822148, 2.073835755953255],
     [1.3218424476669453, 2.280039027652438],
     [1.5305263369319895, 2.583655199859715],
     [1.7367032838294298, 2.8849693173524127],
     [1.9919995812477316, 3.3444991419428254],
     [2.3214874335720737, 3.746609168947945],
     [2.6257068921416016, 4.219627927699429],
     [3.0224730552118966, 4.745172583846427],
     [3.4171493336512118, 5.372911615232418],
     [4.00267938389665, 5.997311113055691],
     [4.538665507059544, 6.7633217931572105],
     [5.0964537080461465, 7.542017709980469],
     [5.937612102490718, 8.494582767235762],
     [6.855955622501851, 9.465459888531576],
     [7.776469488501846, 10.674793278479424],
     [8.85722371899465, 11.916875740681101],
     [10.134169885722144, 13.330838742638255],
     [11.55301986032173, 14.81170883280447],
     [13.137714254870406, 16.54529484241963],
     [15.019887947542076, 18.500956875818567],
     [16.986889411640878, 20.691361812720977],
     [19.358929330898555, 22.821884411658985],
     [21.962699506010424, 25.42774379027305],
     [24.91493545438838, 28.207227158173836],
     [28.177708571764576, 31.199775973178685],
     [32.05645123474679, 34.72688208013862],
     [36.04623543867291, 38.525329251478354],
     [40.71891764031807, 42.37778182082904],
     [45.8057748580099, 46.740772130647926],
     [51.93586549325193, 51.56138455179142],
     [58.22764499522711, 56.63983653180357],
     [65.56347120373863, 62.33677675741977],
     [73.57817197923043, 68.57667616848664],
     [82.34340070590233, 75.36363454081982],
     [92.13199344184662, 82.59964936018662],
     [103.44589757364463, 90.59364749543288],
     [115.47960650240853, 99.07370254787024],
     [129.17846118445937, 108.23586512394414],
     [143.46632902139083, 118.68435626461516],
     [159.40429791492033, 129.52770946289098],
     [177.40328801046977, 140.84696810777604],
     [196.7452568271417, 153.70676591200004],
     [218.8050339839852, 166.94072610207135],
     [242.6781784864164, 181.54828902437282],
     [268.10939869530546, 196.66326326296206],
     [295.70916373091245, 213.95579066067546],
     [326.30599765970044, 231.45714809158784],
     [359.9834862922868, 250.86153537318583],
     [396.89472597534836, 271.33976509182924],
     [436.24004821059606, 292.7053660898765],
     [478.6111123973041, 316.2003022010982],
     [524.7869443785683, 341.4603007662884],
     [573.415533926074, 367.49739920455755],
     [624.7498584266933, 394.3931886916753],
     [678.4040760633584, 423.7487423139537],
     [731.7587522246671, 454.50128566344097],
     [783.8865000181952, 486.54107903863917],
     [835.0950426589582, 522.1659637266313],
     [889.9856943808636, 557.4486314780086],
     [916.7359926229606, 594.2684960149579],
     [915.5597026328171, 629.7712585389686],
     [915.8563950661595, 665.8429064110084],
     [912.7823333803567, 698.8053374306647],
     [904.5242512584814, 725.9840369686002],
     [898.1285029019759, 744.0071307176622],
     [895.8887163024597, 757.1590596041043],
     [897.7148701295298, 764.3567861840843],
     [900.6728634570543, 780.359298526032],
     [903.8336368037496, 808.8198926985237],
     [907.3958646795677, 807.389436682497],
     [910.7468644380027, 798.6830745769123],
     [913.6875493046704, 794.3874039438163],
     [916.3010860325301, 791.9018876915757],
     [918.6127712750178, 792.7510634242733],
     [919.8993041859983, 794.7345001431802],
     [920.3271098260825, 796.9967888858507],
     [920.4630004484947, 799.4081561415728],
     [np.nan, 802.1297572892993],
     [np.nan, 804.8229857019243],
     [np.nan, 807.3226588229511]]).T

N_PE_ERR = np.array([[np.nan, 0.008774341460186297],
                     [0.00615409248744625, 0.009034394295824144],
                     [0.0062949816858216545, 0.009220084781781268],
                     [0.006376167629651708, 0.009209813226194463],
                     [0.006544242388447419, 0.009466371810717578],
                     [0.0064938149722142124, 0.00964873842561198],
                     [0.0067415059586724, 0.009598335194998642],
                     [0.006827603775052593, 0.009841249009396424],
                     [0.006970320293915061, 0.010035849959170395],
                     [0.006868172066514827, 0.010126075266842671],
                     [0.0070122031683123864, 0.010177202561939436],
                     [0.0072545073566464835, 0.010504625766889208],
                     [0.0073484584059599345, 0.01073909244983523],
                     [0.007430658056061912, 0.01079830872262788],
                     [0.007481067467690006, 0.011074215925225228],
                     [0.007673071465402526, 0.011253061456360958],
                     [0.007725813401337267, 0.01132379316422677],
                     [0.007827721405480326, 0.01148738069537375],
                     [0.007847728756052885, 0.01179941881291452],
                     [0.008079649610824913, 0.011847710558694602],
                     [0.008290391063305314, 0.012150385549618947],
                     [0.008875184137899783, 0.01326397821471803],
                     [0.009718162992257473, 0.014556918932732255],
                     [0.010518427633259608, 0.015741937478853774],
                     [0.012430140128730005, 0.018675488901024107],
                     [0.013405248952863769, 0.020817584400078193],
                     [0.014945369827189303, 0.022416860235494385],
                     [0.01656890384269094, 0.024821346565272462],
                     [0.018214996970832065, 0.027035998614362233],
                     [0.02024443811086374, 0.03047625250834507],
                     [0.022846423167836694, 0.03363410202972261],
                     [0.0251706780718719, 0.037213934437225316],
                     [0.02811721099908726, 0.04127900511541727],
                     [0.031213549662685525, 0.045888071686186915],
                     [0.035485857562312706, 0.02627087104686776],
                     [0.03955529519839107, 0.02794871685796796],
                     [0.044056135295145626, 0.02964945121916296],
                     [0.026148299183031742, 0.03164036921746316],
                     [0.028107890870791685, 0.03333511546013579],
                     [0.030102914291065375, 0.035415776681588795],
                     [0.032068618394557546, 0.037478592558340296],
                     [0.03443670660093545, 0.039579392770081157],
                     [0.03683409067220733, 0.04180096060589289],
                     [0.039221422074721346, 0.044347920939525665],
                     [0.041905114166318924, 0.046816171130043216],
                     [0.04481621651123646, 0.049719426995910254],
                     [0.04790185512649714, 0.05235840868506969],
                     [0.05107741527342213, 0.055113853583909744],
                     [0.05456237894323124, 0.05821641619261975],
                     [0.058183596334021814, 0.061436015131697275],
                     [0.06213774031888164, 0.0648076607700041],
                     [0.06574237076394951, 0.06842752654780071],
                     [0.07000052558314707, 0.07181015898141396],
                     [0.07456405853456616, 0.0755653296129104],
                     [0.07913263748653065, 0.07969066030036842],
                     [0.08464643773439917, 0.08357497107019185],
                     [0.09010415756478096, 0.08756148986387302],
                     [0.09567281532554972, 0.09224575610668495],
                     [0.10152285803169292, 0.09683653706063211],
                     [0.10715045189878936, 0.1016300664096832],
                     [0.11422222356731027, 0.10683783925671975],
                     [0.12031688857017286, 0.11189839666820234],
                     [0.12755469516871187, 0.11779400664094908],
                     [0.13584055211350687, 0.12309971480402027],
                     [0.14298790547061913, 0.12864328320461027],
                     [0.15178378898937694, 0.1353001953522579],
                     [0.16017308195705482, 0.14035970053157598],
                     [0.17007276161454854, 0.1473578957485273],
                     [0.17958583259365923, 0.1537522691325961],
                     [0.18885768855898277, 0.16061248852132337],
                     [0.19996790155352073, 0.16796523602181423],
                     [0.20915314974701005, 0.17449769677455151],
                     [0.22111049479718758, 0.18236489841642367],
                     [0.2336259914262655, 0.1918965393055032],
                     [0.24618450941687797, 0.19990888482885794],
                     [0.25896279819227175, 0.207657094184583],
                     [0.2734131503827939, 0.21749659436977709],
                     [0.286325409954145, 0.2255246364449306],
                     [0.30197971207854835, 0.23299653725365488],
                     [0.3253042440766194, 0.24303764382517556],
                     [0.3439649300332235, 0.2530256267050959],
                     [0.36478979285863034, 0.2622899105549834],
                     [0.37042379163051464, 0.2730079111270811],
                     [0.4901852132035742, 0.28358286557630663],
                     [0.3865562882457425, 0.2978747995210256],
                     [0.40003224471132626, 0.31677798230305143],
                     [0.3998994635477402, 0.3416277360298068],
                     [0.4296243066170291, 0.3860853248518765],
                     [0.4504612797336449, 0.45066915948359565],
                     [0.4185439614709594, 0.49177370566758327],
                     [0.38124079292481383, 0.515817723616351],
                     [0.3540935592898222, 0.4393343252840509],
                     [0.3490967051703251, 0.5062299850268914],
                     [0.3477740314643256, 0.4317302044483995],
                     [0.3490123542976562, 0.44024079358308654],
                     [0.34876074353582, 0.4135722521705816],
                     [0.3463551363718693, 0.3935527334480753],
                     [0.3440088234329437, 0.37117975837082895],
                     [0.34159756652604756, 0.35044307715270406],
                     [0.34448299750897604, 0.3382603601249343],
                     [0.5428764152648569, 0.3375955027251507],
                     [3.0116187478582788, 0.33781483906301446],
                     [np.nan, 0.33903562073686544],
                     [np.nan, 0.33672652795030444],
                     [np.nan, 0.33775348890333134]]).T


def test_create():

    ac_led = ACLED(AC_LEVELS, N_PE, N_PE_ERR)

    np.testing.assert_array_equal(ac_led.x, AC_LEVELS)
    np.testing.assert_array_equal(ac_led.y, N_PE)
    np.testing.assert_array_equal(ac_led.y_err, N_PE_ERR)


def test_create_without_error():

    ac_led = ACLED(AC_LEVELS, N_PE)

    np.testing.assert_array_equal(ac_led.x, AC_LEVELS)
    np.testing.assert_array_equal(ac_led.y, N_PE)


def test_call():

    ac_led = ACLED(AC_LEVELS, N_PE, N_PE_ERR)

    x = np.linspace(0, 1000, num=1001)
    y = ac_led(x)

    assert len(y) == len(N_PE)
    assert y.shape[1] == len(x)


def test_1d_y_array():

    ac_led = ACLED(AC_LEVELS, N_PE[0])

    assert len(N_PE[0].shape) == 1
    assert len(ac_led.y.shape) > 1


def test_load_save():

    ac_led = ACLED(AC_LEVELS, N_PE, N_PE_ERR)

    with tempfile.NamedTemporaryFile(suffix='.fits') as f:

        ac_led.save(f.name)
        loaded_ac_led = ACLED.load(f.name)

        assert (ac_led.x == loaded_ac_led.x).all()
        np.testing.assert_array_equal(ac_led.y, loaded_ac_led.y)
        np.testing.assert_array_equal(ac_led.y_err, loaded_ac_led.y_err)


def test_get_item():

    ac_led = ACLED(AC_LEVELS, N_PE, N_PE_ERR)

    ac_led_0 = ac_led[0]

    np.testing.assert_array_equal(ac_led.x, ac_led_0.x)
    np.testing.assert_array_equal(ac_led.y[0].ravel(),
                                  ac_led_0.y.ravel())
    np.testing.assert_array_equal(ac_led.y_err[0].ravel(),
                                  ac_led_0.y_err.ravel())


def test_plot():

    ac_led = ACLED(AC_LEVELS, N_PE, N_PE_ERR)
    ac_led.plot()