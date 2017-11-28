#!/bin/env python
# this code should run in python3.
# Zfits/protobuf loader.
# import protozfitsreader
import numpy
import rawzfitsreader


pixel_remap = [425, 461, 353, 389, 352, 388, 424, 460, 315, 351, 387, 423, 281, 317, 209, 245, 208, 244, 280, 316, 175,
               207, 243, 279, 350, 386, 278, 314, 277, 313, 349, 385, 240, 276, 312, 348, 206, 242, 144, 174, 143, 173,
               205, 241, 116, 142, 172, 204, 713, 749, 641, 677, 640, 676, 712, 748, 603, 639, 675, 711, 569, 605, 497,
               533, 496, 532, 568, 604, 459, 495, 531, 567, 638, 674, 566, 602, 565, 601, 637, 673, 528, 564, 600, 636,
               494, 530, 422, 458, 421, 457, 493, 529, 384, 420, 456, 492, 1001, 1037, 929, 965, 928, 964, 1000, 1036,
               891, 927, 963, 999, 857, 893, 785, 821, 784, 820, 856, 892, 747, 783, 819, 855, 926, 962, 854, 890, 853,
               889, 925, 961, 816, 852, 888, 924, 782, 818, 710, 746, 709, 745, 781, 817, 672, 708, 744, 780, 275, 311,
               203, 239, 202, 238, 274, 310, 169, 201, 237, 273, 141, 171, 91, 115, 90, 114, 140, 170, 69, 89, 113, 139,
               200, 236, 138, 168, 137, 167, 199, 235, 110, 136, 166, 198, 88, 112, 50, 68, 49, 67, 87, 111, 34, 48, 66,
               86, 563, 599, 491, 527, 490, 526, 562, 598, 453, 489, 525, 561, 419, 455, 347, 383, 346, 382, 418, 454,
               309, 345, 381, 417, 488, 524, 416, 452, 415, 451, 487, 523, 378, 414, 450, 486, 344, 380, 272, 308, 271,
               307, 343, 379, 234, 270, 306, 342, 851, 887, 779, 815, 778, 814, 850, 886, 741, 777, 813, 849, 707, 743,
               635, 671, 634, 670, 706, 742, 597, 633, 669, 705, 776, 812, 704, 740, 703, 739, 775, 811, 666, 702, 738,
               774, 632, 668, 560, 596, 559, 595, 631, 667, 522, 558, 594, 630, 135, 165, 85, 109, 84, 108, 134, 164,
               63, 83, 107, 133, 47, 65, 21, 33, 20, 32, 46, 64, 11, 19, 31, 45, 82, 106, 44, 62, 43, 61, 81, 105, 28,
               42, 60, 80, 18, 30, 4, 10, 3, 9, 17, 29, 0, 2, 8, 16, 413, 449, 341, 377, 340, 376, 412, 448, 303, 339,
               375, 411, 269, 305, 197, 233, 196, 232, 268, 304, 163, 195, 231, 267, 338, 374, 266, 302, 265, 301, 337,
               373, 228, 264, 300, 336, 194, 230, 132, 162, 131, 161, 193, 229, 104, 130, 160, 192, 701, 737, 629, 665,
               628, 664, 700, 736, 591, 627, 663, 699, 557, 593, 485, 521, 484, 520, 556, 592, 447, 483, 519, 555, 626,
               662, 554, 590, 553, 589, 625, 661, 516, 552, 588, 624, 482, 518, 410, 446, 409, 445, 481, 517, 372, 408,
               444, 480, 1271, 1270, 1282, 1281, 1273, 1272, 1259, 1258, 1274, 1261, 1260, 1244, 1290, 1289, 1295, 1294,
               1292, 1291, 1284, 1283, 1293, 1286, 1285, 1275, 1246, 1245, 1263, 1262, 1248, 1247, 1228, 1227, 1249,
               1230, 1229, 1207, 1277, 1276, 1288, 1287, 1279, 1278, 1265, 1264, 1280, 1267, 1266, 1250, 1197, 1196,
               1220, 1219, 1199, 1198, 1173, 1172, 1200, 1175, 1174, 1146, 1240, 1239, 1257, 1256, 1242, 1241, 1222,
               1221, 1243, 1224, 1223, 1201, 1148, 1147, 1177, 1176, 1150, 1149, 1118, 1117, 1151, 1120, 1119, 1085,
               1203, 1202, 1226, 1225, 1205, 1204, 1179, 1178, 1206, 1181, 1180, 1152, 1075, 1074, 1110, 1109, 1077,
               1076, 1041, 1040, 1078, 1043, 1042, 1005, 1142, 1141, 1171, 1170, 1144, 1143, 1112, 1111, 1145, 1114,
               1113, 1079, 1007, 1006, 1045, 1044, 1009, 1008, 972, 971, 1010, 974, 973, 936, 1081, 1080, 1116, 1115,
               1083, 1082, 1047, 1046, 1084, 1049, 1048, 1011, 1209, 1208, 1232, 1231, 1211, 1210, 1185, 1184, 1212,
               1187, 1186, 1158, 1252, 1251, 1269, 1268, 1254, 1253, 1234, 1233, 1255, 1236, 1235, 1213, 1160, 1159,
               1189, 1188, 1162, 1161, 1130, 1129, 1163, 1132, 1131, 1097, 1215, 1214, 1238, 1237, 1217, 1216, 1191,
               1190, 1218, 1193, 1192, 1164, 1087, 1086, 1122, 1121, 1089, 1088, 1053, 1052, 1090, 1055, 1054, 1017,
               1154, 1153, 1183, 1182, 1156, 1155, 1124, 1123, 1157, 1126, 1125, 1091, 1019, 1018, 1057, 1056, 1021,
               1020, 984, 983, 1022, 986, 985, 948, 1093, 1092, 1128, 1127, 1095, 1094, 1059, 1058, 1096, 1061, 1060,
               1023, 938, 937, 976, 975, 940, 939, 903, 902, 941, 905, 904, 867, 1013, 1012, 1051, 1050, 1015, 1014,
               978, 977, 1016, 980, 979, 942, 869, 868, 907, 906, 871, 870, 834, 833, 872, 836, 835, 798, 944, 943, 982,
               981, 946, 945, 909, 908, 947, 911, 910, 873, 1099, 1098, 1134, 1133, 1101, 1100, 1065, 1064, 1102, 1067,
               1066, 1029, 1166, 1165, 1195, 1194, 1168, 1167, 1136, 1135, 1169, 1138, 1137, 1103, 1031, 1030, 1069,
               1068, 1033, 1032, 996, 995, 1034, 998, 997, 960, 1105, 1104, 1140, 1139, 1107, 1106, 1071, 1070, 1108,
               1073, 1072, 1035, 950, 949, 988, 987, 952, 951, 915, 914, 953, 917, 916, 879, 1025, 1024, 1063, 1062,
               1027, 1026, 990, 989, 1028, 992, 991, 954, 881, 880, 919, 918, 883, 882, 846, 845, 884, 848, 847, 810,
               956, 955, 994, 993, 958, 957, 921, 920, 959, 923, 922, 885, 800, 799, 838, 837, 802, 801, 765, 764, 803,
               767, 766, 729, 875, 874, 913, 912, 877, 876, 840, 839, 878, 842, 841, 804, 731, 730, 769, 768, 733, 732,
               696, 695, 734, 698, 697, 660, 806, 805, 844, 843, 808, 807, 771, 770, 809, 773, 772, 735, 146, 117, 177,
               145, 213, 178, 179, 147, 249, 250, 214, 215, 211, 176, 246, 210, 282, 247, 248, 212, 318, 319, 283, 284,
               286, 251, 321, 285, 357, 322, 323, 287, 393, 394, 358, 359, 355, 320, 390, 354, 426, 391, 392, 356, 462,
               463, 427, 428, 52, 35, 71, 51, 95, 72, 73, 53, 121, 122, 96, 97, 93, 70, 118, 92, 148, 119, 120, 94, 180,
               181, 149, 150, 152, 123, 183, 151, 219, 184, 185, 153, 255, 256, 220, 221, 217, 182, 252, 216, 288, 253,
               254, 218, 324, 325, 289, 290, 6, 1, 13, 5, 25, 14, 15, 7, 39, 40, 26, 27, 23, 12, 36, 22, 54, 37, 38, 24,
               74, 75, 55, 56, 58, 41, 77, 57, 101, 78, 79, 59, 127, 128, 102, 103, 99, 76, 124, 98, 154, 125, 126, 100,
               186, 187, 155, 156, 430, 395, 465, 429, 501, 466, 467, 431, 537, 538, 502, 503, 499, 464, 534, 498, 570,
               535, 536, 500, 606, 607, 571, 572, 574, 539, 609, 573, 645, 610, 611, 575, 681, 682, 646, 647, 643, 608,
               678, 642, 714, 679, 680, 644, 750, 751, 715, 716, 292, 257, 327, 291, 363, 328, 329, 293, 399, 400, 364,
               365, 361, 326, 396, 360, 432, 397, 398, 362, 468, 469, 433, 434, 436, 401, 471, 435, 507, 472, 473, 437,
               543, 544, 508, 509, 505, 470, 540, 504, 576, 541, 542, 506, 612, 613, 577, 578, 158, 129, 189, 157, 225,
               190, 191, 159, 261, 262, 226, 227, 223, 188, 258, 222, 294, 259, 260, 224, 330, 331, 295, 296, 298, 263,
               333, 297, 369, 334, 335, 299, 405, 406, 370, 371, 367, 332, 402, 366, 438, 403, 404, 368, 474, 475, 439,
               440, 718, 683, 753, 717, 789, 754, 755, 719, 825, 826, 790, 791, 787, 752, 822, 786, 858, 823, 824, 788,
               894, 895, 859, 860, 862, 827, 897, 861, 933, 898, 899, 863, 969, 970, 934, 935, 931, 896, 966, 930, 1002,
               967, 968, 932, 1038, 1039, 1003, 1004, 580, 545, 615, 579, 651, 616, 617, 581, 687, 688, 652, 653, 649,
               614, 684, 648, 720, 685, 686, 650, 756, 757, 721, 722, 724, 689, 759, 723, 795, 760, 761, 725, 831, 832,
               796, 797, 793, 758, 828, 792, 864, 829, 830, 794, 900, 901, 865, 866, 442, 407, 477, 441, 513, 478, 479,
               443, 549, 550, 514, 515, 511, 476, 546, 510, 582, 547, 548, 512, 618, 619, 583, 584, 586, 551, 621, 585,
               657, 622, 623, 587, 693, 694, 658, 659, 655, 620, 690, 654, 726, 691, 692, 656, 762, 763, 727, 728]


class ZFile(object):

    def __init__(self, fname):
        # Save filename
        self.fname = fname
        # Read non-iterable tables (header), keep their contents in memory.
        # self.read_runheader()
        self.patch_id_input = [204, 216, 180, 192, 229, 241, 205, 217, 254, 266, 230, 242, 279,
                               291, 255, 267, 304, 316, 280, 292, 329, 341, 305, 317, 156, 168,
                               132, 144, 181, 193, 157, 169, 206, 218, 182, 194, 231, 243, 207,
                               219, 256, 268, 232, 244, 281, 293, 257, 269, 108, 120, 84, 96,
                               133, 145, 109, 121, 158, 170, 134, 146, 183, 195, 159, 171, 208,
                               220, 184, 196, 233, 245, 209, 221, 60, 72, 40, 50, 85, 97,
                               61, 73, 110, 122, 86, 98, 135, 147, 111, 123, 160, 172, 136,
                               148, 185, 197, 161, 173, 24, 32, 12, 18, 41, 51, 25, 33,
                               62, 74, 42, 52, 87, 99, 63, 75, 112, 124, 88, 100, 137,
                               149, 113, 125, 4, 8, 0, 2, 13, 19, 5, 9, 26, 34,
                               14, 20, 43, 53, 27, 35, 64, 76, 44, 54, 89, 101, 65,
                               77, 228, 239, 240, 252, 251, 262, 263, 275, 274, 285, 286, 298,
                               297, 308, 309, 321, 320, 331, 332, 344, 343, 354, 355, 366, 253,
                               264, 265, 277, 276, 287, 288, 300, 299, 310, 311, 323, 322, 333,
                               334, 346, 345, 356, 357, 368, 367, 377, 378, 387, 278, 289, 290,
                               302, 301, 312, 313, 325, 324, 335, 336, 348, 347, 358, 359, 370,
                               369, 379, 380, 389, 388, 396, 397, 404, 303, 314, 315, 327, 326,
                               337, 338, 350, 349, 360, 361, 372, 371, 381, 382, 391, 390, 398,
                               399, 406, 405, 411, 412, 417, 328, 339, 340, 352, 351, 362, 363,
                               374, 373, 383, 384, 393, 392, 400, 401, 408, 407, 413, 414, 419,
                               418, 422, 423, 426, 353, 364, 365, 376, 375, 385, 386, 395, 394,
                               402, 403, 410, 409, 415, 416, 421, 420, 424, 425, 428, 427, 429,
                               430, 431, 215, 191, 227, 203, 167, 143, 179, 155, 119, 95, 131,
                               107, 71, 49, 83, 59, 31, 17, 39, 23, 7, 1, 11, 3,
                               238, 214, 250, 226, 190, 166, 202, 178, 142, 118, 154, 130, 94,
                               70, 106, 82, 48, 30, 58, 38, 16, 6, 22, 10, 261, 237,
                               273, 249, 213, 189, 225, 201, 165, 141, 177, 153, 117, 93, 129,
                               105, 69, 47, 81, 57, 29, 15, 37, 21, 284, 260, 296, 272,
                               236, 212, 248, 224, 188, 164, 200, 176, 140, 116, 152, 128, 92,
                               68, 104, 80, 46, 28, 56, 36, 307, 283, 319, 295, 259, 235,
                               271, 247, 211, 187, 223, 199, 163, 139, 175, 151, 115, 91, 127,
                               103, 67, 45, 79, 55, 330, 306, 342, 318, 282, 258, 294, 270,
                               234, 210, 246, 222, 186, 162, 198, 174, 138, 114, 150, 126, 90,
                               66, 102, 78]

        self.patch_id_output = [204, 216, 229, 241, 254, 266, 279, 291, 304, 316, 329, 341, 180, 192, 205, 217, 230,
                                242, 255, 267, 280, 292, 305, 317, 156, 168, 181, 193, 206, 218, 231, 243, 256, 268,
                                281, 293, 132, 144, 157, 169, 182, 194, 207, 219, 232, 244, 257, 269, 108, 120, 133,
                                145, 158, 170, 183, 195, 208, 220, 233, 245, 84, 96, 109, 121, 134, 146, 159, 171, 184,
                                196, 209, 221, 60, 72, 85, 97, 110, 122, 135, 147, 160, 172, 185, 197, 40, 50, 61, 73,
                                86, 98, 111, 123, 136, 148, 161, 173, 24, 32, 41, 51, 62, 74, 87, 99, 112, 124, 137,
                                149, 12, 18, 25, 33, 42, 52, 63, 75, 88, 100, 113, 125, 4, 8, 13, 19, 26, 34, 43, 53,
                                64, 76, 89, 101, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 228, 239, 251, 262, 274,
                                285, 297, 308, 320, 331, 343, 354, 240, 252, 263, 275, 286, 298, 309, 321, 332, 344,
                                355, 366, 253, 264, 276, 287, 299, 310, 322, 333, 345, 356, 367, 377, 265, 277, 288,
                                300, 311, 323, 334, 346, 357, 368, 378, 387, 278, 289, 301, 312, 324, 335, 347, 358,
                                369, 379, 388, 396, 290, 302, 313, 325, 336, 348, 359, 370, 380, 389, 397, 404, 303,
                                314, 326, 337, 349, 360, 371, 381, 390, 398, 405, 411, 315, 327, 338, 350, 361, 372,
                                382, 391, 399, 406, 412, 417, 328, 339, 351, 362, 373, 383, 392, 400, 407, 413, 418,
                                422, 340, 352, 363, 374, 384, 393, 401, 408, 414, 419, 423, 426, 353, 364, 375, 385,
                                394, 402, 409, 415, 420, 424, 427, 429, 365, 376, 386, 395, 403, 410, 416, 421, 425,
                                428, 430, 431, 215, 191, 167, 143, 119, 95, 71, 49, 31, 17, 7, 1, 227, 203, 179, 155,
                                131, 107, 83, 59, 39, 23, 11, 3, 238, 214, 190, 166, 142, 118, 94, 70, 48, 30, 16, 6,
                                250, 226, 202, 178, 154, 130, 106, 82, 58, 38, 22, 10, 261, 237, 213, 189, 165, 141,
                                117, 93, 69, 47, 29, 15, 273, 249, 225, 201, 177, 153, 129, 105, 81, 57, 37, 21, 284,
                                260, 236, 212, 188, 164, 140, 116, 92, 68, 46, 28, 296, 272, 248, 224, 200, 176, 152,
                                128, 104, 80, 56, 36, 307, 283, 259, 235, 211, 187, 163, 139, 115, 91, 67, 45, 319, 295,
                                271, 247, 223, 199, 175, 151, 127, 103, 79, 55, 330, 306, 282, 258, 234, 210, 186, 162,
                                138, 114, 90, 66, 342, 318, 294, 270, 246, 222, 198, 174, 150, 126, 102, 78]


    ### INTERNAL METHODS ###########################################################

    def _read_file(self):
        # Read file. Return a serialized string
        assert self.ttype in ["RunHeader", "Events", "RunTails"]
        rawzfitsreader.open("%s:%s" % (self.fname, self.ttype))

    def _read_message(self):
        # Read next message. Fills property self.rawmessage and self.numrows

        self.rawmessage = rawzfitsreader.readEvent()
        self.numrows = rawzfitsreader.getNumRows()

    def _extract_field(self, obj, field):
        # Read a specific field in object 'obj' given as input 'field'
        if not obj.HasField(field):
            raise Exception("No field %s found in object %s" % (field, str(obj)))
        return (getattr(obj, field))

    def _get_numpyfield(self, field):
        numpyfield = toNumPyArray(field)
        return (numpyfield)

    ### PUBLIC METHODS #############################################################

    def list_tables(self):
        return rawzfitsreader.listAllTables(self.fname)

    def read_runheader(self):
        # Get number of events in file
        import L0_pb2
        try:
            assert (self.ttype == "RunHeader")
        except:
            self.ttype = "RunHeader"
            self._read_file()

        self._read_message()
        self.header = L0_pb2.CameraRunHeader()
        self.header.ParseFromString(self.rawmessage)
        # self.print_listof_fields(self.header)

    def read_event(self):
        # read an event. Assume it is a camera event
        # C++ return a serialized string, python protobuf rebuild message from serial string
        import L0_pb2

        if self.ttype == "Events":
            self.eventnumber += 1
        else:
            self.ttype = "Events"
            self._read_file()
            self.eventnumber = 1

        # print("Reading event number %d" %self.eventnumber)
        self._read_message()

        self.event = L0_pb2.CameraEvent()
        self.event.ParseFromString(self.rawmessage)
        # self.print_listof_fields(self.event)

    def rewind_table(self):
        # Rewind the current reader. Go to the beginning of the table.
        rawzfitsreader.rewindTable()

    def move_to_next_event(self):
        # Iterate over events
        i = 0
        numrows = i + 2
        # Hook to deal with file with no header (1)
        if hasattr(self, 'numrows'):
            numrows = self.numrows
        # End - Hook to deal with file with no header (1)
        while i < numrows:
            self.read_event()
            # Hook to deal with file with no header (2)
            if hasattr(self, 'numrows'):
                numrows = self.numrows
            # End - Hook to deal with file with no header (2)

            # Hook to deal with file with no header (3)
            try:
                run_id = self.get_run_id()
                event_id = self.get_event_id()
            except:
                run_id = 0
                event_id = self.eventnumber
                # print('No header in this file, run_id set to 0 and event_id set to event_number')
            # Hook to deal with file with no header (3)

            yield run_id, event_id
            i += 1

    def get_telescope_id(self):

        return self.event.telescopeID

    def get_event_number(self):

        return self.event.eventNumber

    def get_run_id(self):

        return (self._get_numpyfield(self.header.runNumber))

    def get_central_event_gps_time(self):

        time_second = self.event.trig.timeSec
        time_nanosecond = self.event.trig.timeNanoSec
        return time_second * 1E9 + time_nanosecond

    def get_local_time(self):

        time_second = self.event.local_time_sec
        time_nanosecond = self.event.local_time_nanosec

        return time_second * 1E9 + time_nanosecond

    def get_baseline(self):
        """
        Get the baselines for all channels
        :return: dict() of baselines (value) per pixel indices (key)
        """
        waveforms = self.event.hiGain.waveforms

        try:
            baselines = self._get_numpyfield(waveforms.baselines)

        except:

            n_pixels = self._get_numpyfield(waveforms.pixelsIndices).shape[0]
            baselines = numpy.zeros(n_pixels) * numpy.nan

        pixels = self._get_numpyfield(waveforms.pixelsIndices)
        properties = numpy.array(list(dict(zip(pixels, baselines)).values()))

        return properties

    def get_event_number_array(self):

        return self._get_numpyfield(self.event.arrayEvtNum)

    def get_camera_event_type(self):

        return self.event.event_type

    def get_array_event_type(self):

        return self.event.eventType

    def get_num_channels(self):

        return self._get_numpyfield(self.event.head.numGainChannels)

    def _get_adc(self, channel, telescope_id=None):
        # Expect hi/lo -> Will append Gain at the end -> hiGain/loGain
        sel_channel = self._extract_field(self.event, "%sGain" % channel)
        return (sel_channel)

    def get_pixel_position(self, telescope_id=None):

        return None

    def get_number_of_pixels(self, telescope_id=None):

        n_pixels = self._get_numpyfield(self.event.hiGain.waveforms.pixelsIndices).shape[0]
        return n_pixels

    def get_adcs_samples(self, telescope_id=None):
        """
        Get the samples for all channels

        :param telescope_id: id of the telescopeof interest
        :return: dictionnary of samples (value) per pixel indices (key)
        """
        waveforms = self.event.hiGain.waveforms
        samples = self._get_numpyfield(waveforms.samples)
        pixels = self._get_numpyfield(waveforms.pixelsIndices)
        npixels = len(pixels)
        # Structured array (dict)
        samples = samples.reshape(npixels, -1)
        properties = numpy.array(list(dict(zip(pixels, samples)).values()))
        return properties

    def get_num_samples(self):

        waveforms = self.event.hiGain.waveforms
        samples = self._get_numpyfield(waveforms.samples)
        pixels = self._get_numpyfield(waveforms.pixelsIndices)

        return samples.shape[0] // pixels.shape[0]

    def get_trigger_input_traces(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescopeof interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''

        frames = self._get_numpyfield(self.event.trigger_input_traces)
        frames = frames.reshape(frames.shape[0] // 3, 3)
        frames = frames.reshape(frames.shape[0] // 192, 3, 192)
        frames = frames[..., :144]
        frames = frames.reshape(frames.shape[0], frames.shape[1]*frames.shape[2])
        frames = frames.T
        frames = numpy.array(list(dict(zip(self.patch_id_input, frames)).values()))

        return frames

    def get_trigger_output_patch7(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescope of interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''
        frames = self._get_numpyfield(self.event.trigger_output_patch7)
        n_samples = int(frames.shape[0] / 18 / 3)
        frames = numpy.unpackbits(frames.reshape(n_samples, 3, 18, 1), axis=-1)[..., ::-1].reshape(n_samples, 3,
                                                                                                   144).reshape(
            n_samples, 432).T

        patches = self.patch_id_output
        properties = dict(zip(patches, frames))

        frames = numpy.array(list(properties.values()))

        return frames

    def get_trigger_output_patch19(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescope of interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''
        frames = self._get_numpyfield(self.event.trigger_output_patch19)
        n_samples = int(frames.shape[0] / 18 / 3)
        frames = numpy.unpackbits(frames.reshape(n_samples, 3, 18, 1), axis=-1)[..., ::-1].reshape(n_samples, 3,144).reshape(n_samples, 432).T
        patches = self.patch_id_output
        properties = dict(zip(patches, frames))
        properties = numpy.array(list(properties.values()))

        return properties

    def get_pixel_flags(self, telescope_id=None):
        '''
        Get the flag of pixels
        :param id of the telescopeof interest
        :return: dictionnary of flags (value) per pixel indices (key)
        '''
        waveforms = self.event.hiGain.waveforms
        flags = self._get_numpyfield(self.event.pixels_flags)
        pixels = self._get_numpyfield(waveforms.pixelsIndices)
        npixels = len(pixels)
        properties = numpy.array(list(dict(zip(pixels, flags)).values()), dtype=bool)
        return properties

    def print_listof_fields(self, obj):
        fields = [f.name for f in obj.DESCRIPTOR.fields]
        print(fields)
        return (fields)


# below are utility functions used to convert from AnyArray to numPyArrays
def typeNone(data):
    raise Exception("This any array has no defined type")


def typeS8(data):
    return numpy.fromstring(data, numpy.int8)


def typeU8(data):
    return numpy.fromstring(data, numpy.uint8)


def typeS16(data):
    return numpy.fromstring(data, numpy.int16)


def typeU16(data):
    return numpy.fromstring(data, numpy.uint16)


def typeS32(data):
    return numpy.fromstring(data, numpy.int32)


def typeU32(data):
    return numpy.fromstring(data, numpy.uint32)


def typeS64(data):
    return numpy.fromstring(data, numpy.int64)


def typeU64(data):
    return numpy.fromstring(data, numpy.uint64)


def typeFloat(data):
    return numpy.fromstring(data, numpy.float)


def typeDouble(data):
    return numpy.fromstring(data, numpy.double)


def typeBool(any_array):
    raise Exception("I have no idea if the boolean representation of the anyarray is the same as the numpy one")


type_id_to_converter_map = {
  0: typeNone,
  1: typeS8,
  2: typeU8,
  3: typeS16,
  4: typeU16,
  5: typeS32,
  6: typeU32,
  7: typeS64,
  8: typeU64,
  9: typeFloat,
  10: typeDouble,
  11: typeBool,
}


def toNumPyArray(any_array):
    try:
        return type_id_to_converter_map[any_array.type](any_array.data)
    except Exception as e:
        raise Exception("Conversion to NumpyArray failed with error %s" % e)
