import matplotlib.pyplot as plt

avg = {'aggregation':'fed_avg','bias':'0','byz_type':'no','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0904, 0.0948, 0.0996, 0.0997, 0.1038, 0.1097, 0.1173, 0.1285, 0.1224, 0.1268, 0.1288, 0.1345, 0.1332, 0.1352, 0.1456, 0.1559, 0.164, 0.165, 0.1726, 0.1907, 0.1934, 0.1864, 0.2285, 0.2394, 0.2588, 0.2968, 0.2986, 0.2956, 0.3021, 0.327, 0.3718, 0.3857, 0.3814, 0.372, 0.3609, 0.3418, 0.3703, 0.3656, 0.3732, 0.3844, 0.3655, 0.3536, 0.3738, 0.3919, 0.4083, 0.4395, 0.4476, 0.4683, 0.462, 0.461, 0.479, 0.4912, 0.5026, 0.514, 0.4938, 0.503, 0.4625, 0.4649, 0.5049, 0.5536, 0.558, 0.5425, 0.5704, 0.6103, 0.6196, 0.6323, 0.6101, 0.618, 0.5922, 0.6286, 0.656, 0.65, 0.666, 0.6753, 0.664, 0.6403, 0.6648, 0.7173, 0.7468, 0.6934, 0.6678, 0.6973, 0.728, 0.7449, 0.7302, 0.7006, 0.7539, 0.756, 0.7372, 0.7689, 0.7409, 0.7649, 0.7127, 0.6911, 0.6099, 0.6456, 0.6621, 0.6676, 0.568, 0.6168, 0.6141, 0.7343, 0.7881, 0.7066, 0.6698, 0.6157, 0.7191, 0.7109, 0.7227, 0.766, 0.8292, 0.8354, 0.7979, 0.805, 0.8121, 0.7721, 0.7537, 0.7175, 0.6539, 0.6823, 0.7505, 0.7518, 0.781, 0.763, 0.8282, 0.7432, 0.7839, 0.7478, 0.7918, 0.7728, 0.7722, 0.8168, 0.8319, 0.8039, 0.7966, 0.786, 0.7902, 0.7703, 0.7927, 0.7647, 0.7852, 0.8075, 0.8329, 0.7844, 0.8107, 0.8149, 0.8171, 0.8081, 0.8507, 0.8439, 0.867, 0.8396, 0.8645, 0.8604, 0.8683, 0.8472, 0.7742, 0.8253, 0.827, 0.8605, 0.8457, 0.8316, 0.8353, 0.7962, 0.8151, 0.8668, 0.8708, 0.8765, 0.8456, 0.8685, 0.8647, 0.8814, 0.8681, 0.8764, 0.8608, 0.8758, 0.8521, 0.8649, 0.8093, 0.7885, 0.7483, 0.7599, 0.8062, 0.8554, 0.8826, 0.8534, 0.8785, 0.8776, 0.8886, 0.873, 0.8791, 0.8501, 0.7975, 0.8041, 0.8411, 0.8894, 0.8859, 0.8879, 0.8868, 0.8885, 0.8815, 0.8888, 0.8889, 0.8908, 0.885, 0.8931, 0.8884, 0.8946, 0.8779, 0.8776, 0.8613, 0.888, 0.883, 0.8949, 0.8772, 0.8873, 0.8822, 0.893, 0.8929, 0.8959, 0.8993, 0.8989, 0.8965, 0.8917, 0.8906, 0.864, 0.871, 0.8495, 0.8819, 0.877, 0.8942, 0.8855, 0.893, 0.8928, 0.8814, 0.9028, 0.8923, 0.8848, 0.8723, 0.8895, 0.895, 0.8968, 0.8908, 0.9023, 0.8889, 0.8918, 0.9038, 0.889, 0.9006, 0.8895, 0.9039, 0.9063, 0.9059, 0.9047, 0.9033, 0.9001, 0.897, 0.9043, 0.9015, 0.8901, 0.8779, 0.8856, 0.8775, 0.8563, 0.8986, 0.8888, 0.9089, 0.9118, 0.9114, 0.9119, 0.9116, 0.9073, 0.9095, 0.899, 0.9003, 0.8871, 0.9079, 0.9092, 0.9034, 0.9047, 0.9054, 0.8844, 0.9052, 0.9051, 0.9057, 0.9064, 0.9019, 0.9131, 0.9121, 0.9102, 0.904, 0.8984, 0.8864, 0.8305, 0.8898, 0.9069, 0.9072, 0.9161, 0.9081, 0.9152]}

avg_attack = {'aggregation':'fed_avg','bias':'0','byz_type':'trim_attack','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0923, 0.0921, 0.0938, 0.0949, 0.0975, 0.0996, 0.1008, 0.1013, 0.1021, 0.1019, 0.1028, 0.1036, 0.1026, 0.1029, 0.1029, 0.103, 0.1034, 0.1035, 0.1044, 0.1039, 0.1047, 0.1045, 0.1042, 0.1041, 0.1044, 0.1044, 0.1054, 0.1052, 0.1054, 0.1057, 0.1049, 0.1043, 0.1041, 0.1048, 0.1046, 0.105, 0.1057, 0.1072, 0.108, 0.108, 0.1079, 0.1085, 0.1075, 0.1071, 0.1067, 0.1069, 0.1096, 0.1112, 0.1108, 0.1085, 0.1074, 0.1071, 0.1068, 0.107, 0.1065, 0.1074, 0.1092, 0.1096, 0.11, 0.1107, 0.1107, 0.1129, 0.1118, 0.1124, 0.1126, 0.1125, 0.1112, 0.1136, 0.1122, 0.117, 0.1216, 0.1185, 0.118, 0.1198, 0.1215, 0.1246, 0.1273, 0.1266, 0.1235, 0.1198, 0.1257, 0.1233, 0.1219, 0.1231, 0.1221, 0.1179, 0.1202, 0.1191, 0.1256, 0.1234, 0.1222, 0.1204, 0.1198, 0.1176, 0.1161, 0.1166, 0.1172, 0.1211, 0.1344, 0.129, 0.1336, 0.1443, 0.1648, 0.167, 0.1769, 0.1654, 0.1756, 0.1717, 0.1903, 0.1938, 0.2047, 0.2095, 0.2384, 0.2421, 0.2394, 0.2992, 0.349, 0.3513, 0.3406, 0.3249, 0.3711, 0.3473, 0.3518, 0.372, 0.3546, 0.3575, 0.3801, 0.41, 0.4135, 0.4743, 0.4761, 0.497, 0.4793, 0.4977, 0.5291, 0.5341, 0.5862, 0.5518, 0.5574, 0.5448, 0.5653, 0.6243, 0.5956, 0.6286, 0.6262, 0.6786, 0.6765, 0.6824, 0.6461, 0.676, 0.6894, 0.6888, 0.6886, 0.6767, 0.691, 0.7221, 0.7241, 0.6877, 0.7471, 0.7324, 0.7279, 0.7374, 0.7395, 0.7241, 0.7314, 0.7313, 0.731, 0.7398, 0.7363, 0.7438, 0.7387, 0.7692, 0.7687, 0.7469, 0.7362, 0.7568, 0.7418, 0.7693, 0.7402, 0.6211, 0.6087, 0.6128, 0.58, 0.6329, 0.6577, 0.6952, 0.7172, 0.7224, 0.7074, 0.7653, 0.652, 0.6321, 0.6958, 0.6974, 0.7906, 0.8083, 0.8273, 0.8257, 0.8051, 0.831, 0.8367, 0.7626, 0.6904, 0.6675, 0.67, 0.6806, 0.7734, 0.7508, 0.739, 0.6741, 0.7169, 0.7739, 0.8018, 0.7693, 0.7512, 0.7492, 0.8299, 0.7942, 0.7661, 0.7831, 0.7663, 0.7573, 0.7147, 0.7786, 0.7823, 0.8087, 0.8404, 0.8605, 0.8618, 0.8603, 0.8619, 0.8333, 0.8615, 0.8632, 0.8626, 0.8689, 0.8627, 0.8473, 0.8427, 0.8108, 0.8477, 0.8594, 0.8636, 0.8669, 0.8588, 0.8273, 0.8236, 0.8007, 0.8018, 0.7078, 0.7619, 0.8579, 0.7904, 0.7557, 0.7893, 0.8518, 0.8651, 0.8617, 0.8724, 0.8696, 0.8722, 0.8727, 0.8774, 0.8704, 0.8727, 0.88, 0.8684, 0.8788, 0.8761, 0.8613, 0.8784, 0.8766, 0.8739, 0.8705, 0.8822, 0.8786, 0.8803, 0.8761, 0.8741, 0.8769, 0.8811, 0.8769, 0.876, 0.817, 0.7424, 0.7533, 0.7421, 0.7027, 0.7968, 0.8509, 0.867, 0.8699, 0.8774, 0.8859, 0.8844, 0.8862, 0.8816, 0.8807, 0.8856, 0.8815]}

med = {'aggregation':'fed_med','bias':'0','byz_type':'no','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0844, 0.0916, 0.0968, 0.0995, 0.103, 0.11, 0.1155, 0.1327, 0.1401, 0.1297, 0.143, 0.139, 0.1483, 0.1412, 0.1361, 0.1417, 0.1388, 0.1506, 0.1578, 0.1606, 0.1689, 0.1771, 0.2171, 0.2071, 0.2123, 0.2287, 0.249, 0.2755, 0.3244, 0.311, 0.3172, 0.3471, 0.3607, 0.3655, 0.3643, 0.3863, 0.4141, 0.3644, 0.3867, 0.3996, 0.3967, 0.406, 0.3817, 0.3984, 0.4208, 0.4211, 0.4323, 0.4404, 0.4681, 0.4854, 0.4978, 0.4879, 0.489, 0.4869, 0.5682, 0.529, 0.5071, 0.5378, 0.5684, 0.6019, 0.5817, 0.5691, 0.5994, 0.5773, 0.5805, 0.6024, 0.6212, 0.6288, 0.6458, 0.6485, 0.6431, 0.6536, 0.6562, 0.655, 0.708, 0.7076, 0.7068, 0.7089, 0.6874, 0.6934, 0.7083, 0.7457, 0.7064, 0.6548, 0.7382, 0.6756, 0.7352, 0.7094, 0.6666, 0.6994, 0.6972, 0.6561, 0.7078, 0.662, 0.5806, 0.613, 0.6413, 0.6128, 0.6683, 0.6551, 0.6955, 0.7386, 0.8023, 0.7458, 0.7176, 0.7135, 0.7714, 0.7366, 0.7104, 0.7171, 0.722, 0.718, 0.7454, 0.7242, 0.7213, 0.7698, 0.8102, 0.7604, 0.8178, 0.7525, 0.6298, 0.6801, 0.7191, 0.7552, 0.823, 0.8284, 0.8018, 0.8121, 0.7425, 0.7911, 0.7378, 0.7057, 0.7607, 0.8004, 0.7996, 0.8006, 0.723, 0.784, 0.8361, 0.8464, 0.7912, 0.8006, 0.7382, 0.7752, 0.8176, 0.8508, 0.8558, 0.8561, 0.8436, 0.8359, 0.8544, 0.8406, 0.8284, 0.8586, 0.8591, 0.8691, 0.8276, 0.8423, 0.8035, 0.8488, 0.8303, 0.821, 0.8378, 0.8389, 0.8069, 0.7923, 0.8144, 0.7645, 0.7873, 0.8329, 0.8492, 0.8735, 0.8808, 0.8691, 0.8725, 0.8667, 0.8751, 0.8733, 0.8689, 0.8644, 0.8738, 0.8812, 0.8697, 0.8605, 0.8837, 0.8775, 0.8527, 0.8478, 0.86, 0.8774, 0.8529, 0.8778, 0.8682, 0.8619, 0.8601, 0.8575, 0.84, 0.8606, 0.8227, 0.8626, 0.8526, 0.8728, 0.8777, 0.8898, 0.8806, 0.8876, 0.8833, 0.8798, 0.8689, 0.8785, 0.8833, 0.8923, 0.8958, 0.8803, 0.8816, 0.8872, 0.885, 0.8917, 0.8793, 0.8556, 0.7958, 0.7859, 0.8043, 0.8688, 0.887, 0.8877, 0.8855, 0.8611, 0.8722, 0.8493, 0.8905, 0.8866, 0.8945, 0.8969, 0.8839, 0.8604, 0.8703, 0.8964, 0.898, 0.8955, 0.891, 0.8921, 0.8867, 0.8929, 0.8771, 0.8614, 0.8876, 0.8959, 0.8953, 0.8946, 0.8853, 0.896, 0.9006, 0.9008, 0.9007, 0.8977, 0.8933, 0.8872, 0.8963, 0.9001, 0.9009, 0.901, 0.8968, 0.8938, 0.8883, 0.8823, 0.8857, 0.8759, 0.8916, 0.8947, 0.9026, 0.8962, 0.9045, 0.9033, 0.902, 0.8957, 0.9036, 0.8911, 0.9014, 0.9015, 0.9042, 0.8978, 0.8949, 0.8896, 0.885, 0.8727, 0.899, 0.8988, 0.9046, 0.9034, 0.9065, 0.9016, 0.8879, 0.9078, 0.908, 0.9032, 0.9043, 0.9079, 0.8984, 0.8817]}

med_attack = {'aggregation':'fed_med','bias':'0','byz_type':'trim_attack','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0957, 0.0986, 0.0995, 0.102, 0.1022, 0.1028, 0.1029, 0.1035, 0.1033, 0.1043, 0.1039, 0.104, 0.1038, 0.1036, 0.1035, 0.1036, 0.1037, 0.1038, 0.1038, 0.1037, 0.1037, 0.1036, 0.1035, 0.1034, 0.1035, 0.1035, 0.1034, 0.1034, 0.1032, 0.1032, 0.1033, 0.1033, 0.1035, 0.1034, 0.1043, 0.1042, 0.1046, 0.105, 0.1038, 0.1036, 0.1038, 0.1038, 0.1037, 0.1037, 0.1048, 0.1043, 0.1053, 0.1055, 0.1062, 0.1063, 0.1062, 0.1082, 0.1068, 0.1078, 0.1091, 0.1088, 0.1095, 0.1096, 0.1097, 0.1086, 0.1088, 0.1086, 0.1077, 0.1077, 0.1076, 0.1071, 0.1114, 0.1109, 0.1091, 0.108, 0.1079, 0.1095, 0.1089, 0.1117, 0.1138, 0.118, 0.123, 0.123, 0.1323, 0.1406, 0.1417, 0.1446, 0.1564, 0.1448, 0.1413, 0.1498, 0.1524, 0.1495, 0.1612, 0.1717, 0.208, 0.2267, 0.2556, 0.2685, 0.2698, 0.2634, 0.2707, 0.2825, 0.3234, 0.3565, 0.3728, 0.3848, 0.405, 0.3929, 0.4215, 0.3988, 0.3927, 0.4112, 0.4296, 0.4503, 0.4715, 0.4494, 0.5457, 0.5549, 0.516, 0.532, 0.5212, 0.4991, 0.5466, 0.6089, 0.6487, 0.662, 0.6713, 0.6612, 0.6885, 0.6985, 0.6451, 0.6406, 0.6387, 0.6474, 0.6414, 0.6934, 0.6954, 0.6981, 0.6957, 0.7168, 0.7348, 0.713, 0.7227, 0.7441, 0.7464, 0.7497, 0.7122, 0.7265, 0.7482, 0.7569, 0.7659, 0.7694, 0.7458, 0.7234, 0.7543, 0.7768, 0.7692, 0.7603, 0.7748, 0.7624, 0.7487, 0.7044, 0.6822, 0.5825, 0.6232, 0.5754, 0.6414, 0.707, 0.7982, 0.8055, 0.8014, 0.7723, 0.6505, 0.6867, 0.624, 0.4536, 0.5711, 0.5908, 0.621, 0.7206, 0.7693, 0.7962, 0.8235, 0.8248, 0.8281, 0.808, 0.8078, 0.8058, 0.7386, 0.7492, 0.7718, 0.7916, 0.834, 0.8408, 0.8283, 0.8171, 0.8315, 0.8279, 0.8293, 0.8208, 0.8462, 0.8438, 0.8496, 0.8448, 0.8075, 0.7552, 0.7032, 0.7064, 0.727, 0.7746, 0.8333, 0.8236, 0.8318, 0.8471, 0.8556, 0.8554, 0.837, 0.8224, 0.7549, 0.7679, 0.6869, 0.6865, 0.7232, 0.8174, 0.8558, 0.8519, 0.8525, 0.8589, 0.864, 0.8545, 0.8515, 0.8454, 0.8407, 0.8467, 0.8528, 0.8427, 0.828, 0.8536, 0.8651, 0.8616, 0.861, 0.8541, 0.8639, 0.8569, 0.8676, 0.8629, 0.8543, 0.8692, 0.829, 0.8412, 0.784, 0.8101, 0.8662, 0.8684, 0.8667, 0.868, 0.8519, 0.8339, 0.8161, 0.8056, 0.7914, 0.7654, 0.7639, 0.8113, 0.794, 0.854, 0.8582, 0.8618, 0.8724, 0.8698, 0.8697, 0.8786, 0.8773, 0.8764, 0.8748, 0.8762, 0.8759, 0.8614, 0.8591, 0.8389, 0.8336, 0.7312, 0.7649, 0.8088, 0.8151, 0.8786, 0.8817, 0.885, 0.88, 0.8666, 0.8824, 0.8828, 0.8799, 0.877, 0.8751, 0.8858, 0.8858, 0.8836, 0.8736, 0.8759, 0.8737, 0.8788, 0.8798, 0.8873]}

krum = {'aggregation':'krum','bias':'0','byz_type':'no','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0948, 0.0881, 0.085, 0.0944, 0.106, 0.1217, 0.1516, 0.1691, 0.1794, 0.1943, 0.1877, 0.2079, 0.2203, 0.2586, 0.2584, 0.2375, 0.2239, 0.2366, 0.2792, 0.2788, 0.2832, 0.2633, 0.2932, 0.2758, 0.2904, 0.2952, 0.2853, 0.2562, 0.2951, 0.293, 0.279, 0.329, 0.3163, 0.2947, 0.3021, 0.3525, 0.3686, 0.3822, 0.4071, 0.4258, 0.4152, 0.4186, 0.4494, 0.4421, 0.4683, 0.4923, 0.513, 0.5408, 0.5618, 0.5329, 0.5324, 0.5118, 0.5771, 0.5699, 0.5559, 0.5322, 0.5213, 0.5082, 0.562, 0.5712, 0.5801, 0.549, 0.5582, 0.5951, 0.5787, 0.5868, 0.5275, 0.6065, 0.6018, 0.5994, 0.6031, 0.5961, 0.6386, 0.6197, 0.6328, 0.6027, 0.687, 0.6218, 0.704, 0.713, 0.6908, 0.7339, 0.6287, 0.6737, 0.7269, 0.7079, 0.7651, 0.5978, 0.7021, 0.6479, 0.6038, 0.6406, 0.7045, 0.6143, 0.5154, 0.6577, 0.7029, 0.692, 0.7153, 0.7914, 0.7192, 0.642, 0.6149, 0.7578, 0.7489, 0.7636, 0.7695, 0.7326, 0.574, 0.6771, 0.751, 0.6673, 0.7274, 0.7136, 0.6522, 0.7365, 0.6146, 0.7347, 0.7046, 0.7581, 0.661, 0.6747, 0.7412, 0.8109, 0.806, 0.7424, 0.8119, 0.8123, 0.7804, 0.7747, 0.7049, 0.6325, 0.773, 0.793, 0.8084, 0.7642, 0.7471, 0.809, 0.7975, 0.8078, 0.8239, 0.8516, 0.8355, 0.8664, 0.8262, 0.8055, 0.8495, 0.8389, 0.8365, 0.8334, 0.7901, 0.7746, 0.7767, 0.8214, 0.849, 0.8364, 0.7487, 0.7816, 0.816, 0.828, 0.7962, 0.8155, 0.7882, 0.7989, 0.8224, 0.8568, 0.7952, 0.8543, 0.844, 0.7919, 0.8138, 0.8579, 0.8608, 0.8534, 0.8061, 0.7738, 0.8093, 0.8524, 0.7976, 0.8035, 0.7499, 0.8033, 0.8674, 0.8693, 0.8411, 0.8745, 0.8841, 0.8674, 0.874, 0.8732, 0.8658, 0.8596, 0.8729, 0.8785, 0.8624, 0.87, 0.872, 0.8838, 0.8735, 0.8542, 0.8498, 0.8669, 0.8749, 0.8206, 0.8283, 0.8825, 0.8786, 0.8764, 0.865, 0.8693, 0.847, 0.885, 0.8748, 0.8684, 0.8841, 0.8596, 0.872, 0.8846, 0.876, 0.8797, 0.8871, 0.8763, 0.8586, 0.8772, 0.8902, 0.8927, 0.8949, 0.876, 0.8851, 0.8878, 0.8878, 0.847, 0.8478, 0.8695, 0.8871, 0.8925, 0.8887, 0.8842, 0.8784, 0.8627, 0.871, 0.8786, 0.8737, 0.892, 0.8821, 0.8317, 0.8414, 0.8753, 0.8777, 0.8898, 0.8883, 0.8786, 0.889, 0.8931, 0.8931, 0.8942, 0.8986, 0.8818, 0.8796, 0.856, 0.8159, 0.8411, 0.8887, 0.9006, 0.8808, 0.8867, 0.8914, 0.885, 0.8754, 0.8404, 0.8582, 0.8503, 0.8763, 0.8887, 0.9, 0.8941, 0.8975, 0.8815, 0.8998, 0.8968, 0.9042, 0.8994, 0.8972, 0.9002, 0.9095, 0.902, 0.8851, 0.8742, 0.8986, 0.8961, 0.8902, 0.8898, 0.8426, 0.8611, 0.8786, 0.8933, 0.9017, 0.8948, 0.905, 0.91]}

krum_attack = {'aggregation':'krum','bias':'0','byz_type':'krum_attack','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0958, 0.1024, 0.1038, 0.1037, 0.1033, 0.1027, 0.1028, 0.1029, 0.1036, 0.1035, 0.1038, 0.1035, 0.1036, 0.1036, 0.104, 0.1042, 0.1057, 0.1061, 0.1052, 0.1075, 0.1117, 0.1119, 0.1182, 0.1202, 0.1245, 0.1163, 0.1309, 0.1233, 0.1326, 0.1418, 0.1693, 0.1827, 0.2195, 0.1673, 0.1791, 0.188, 0.2027, 0.2424, 0.2164, 0.2112, 0.2223, 0.2466, 0.2519, 0.2783, 0.3013, 0.3409, 0.3651, 0.4242, 0.3965, 0.3747, 0.4322, 0.4317, 0.4563, 0.3977, 0.4007, 0.4273, 0.4737, 0.513, 0.5403, 0.4747, 0.4939, 0.5009, 0.5009, 0.5956, 0.58, 0.5686, 0.5246, 0.5635, 0.5142, 0.5937, 0.636, 0.5973, 0.6039, 0.6144, 0.5815, 0.5745, 0.5802, 0.6519, 0.7171, 0.6981, 0.6793, 0.6588, 0.7652, 0.7393, 0.6584, 0.6935, 0.7389, 0.6354, 0.5982, 0.6288, 0.6594, 0.6791, 0.6125, 0.6622, 0.719, 0.606, 0.5625, 0.6596, 0.6679, 0.7074, 0.761, 0.7095, 0.5595, 0.6271, 0.6282, 0.7505, 0.805, 0.7089, 0.6208, 0.6767, 0.6689, 0.7, 0.5844, 0.6096, 0.709, 0.7496, 0.8215, 0.744, 0.7348, 0.6869, 0.7507, 0.8028, 0.8043, 0.8409, 0.8395, 0.812, 0.835, 0.7648, 0.7536, 0.7139, 0.761, 0.7439, 0.8086, 0.7821, 0.6397, 0.659, 0.6849, 0.7844, 0.849, 0.8216, 0.8524, 0.8239, 0.8367, 0.8356, 0.7905, 0.7171, 0.8078, 0.7843, 0.8438, 0.8008, 0.8003, 0.7948, 0.791, 0.8612, 0.8583, 0.8379, 0.8214, 0.8571, 0.8032, 0.8291, 0.8618, 0.8668, 0.8467, 0.7815, 0.7829, 0.8037, 0.808, 0.8457, 0.849, 0.8419, 0.6659, 0.7849, 0.8681, 0.8702, 0.8699, 0.863, 0.8473, 0.8725, 0.8698, 0.8553, 0.8318, 0.8657, 0.8469, 0.8759, 0.8529, 0.8544, 0.8363, 0.8647, 0.872, 0.8504, 0.7859, 0.8279, 0.8404, 0.8611, 0.8688, 0.8483, 0.8739, 0.8756, 0.8623, 0.8612, 0.8613, 0.8529, 0.8789, 0.8611, 0.882, 0.8805, 0.8808, 0.8627, 0.8688, 0.8537, 0.8706, 0.8778, 0.8797, 0.8888, 0.8607, 0.8444, 0.8724, 0.8499, 0.8711, 0.8491, 0.8505, 0.8463, 0.8773, 0.8614, 0.8776, 0.8372, 0.8379, 0.8064, 0.795, 0.8726, 0.8796, 0.8711, 0.8696, 0.8898, 0.8701, 0.8767, 0.8921, 0.8777, 0.875, 0.8771, 0.8965, 0.8879, 0.8854, 0.8644, 0.8878, 0.875, 0.8761, 0.8685, 0.8831, 0.8815, 0.8955, 0.8684, 0.8685, 0.8775, 0.8956, 0.8768, 0.88, 0.8964, 0.8765, 0.9037, 0.9024, 0.899, 0.8844, 0.8841, 0.8988, 0.8933, 0.8794, 0.8611, 0.8493, 0.8847, 0.8974, 0.8496, 0.8896, 0.8976, 0.8766, 0.894, 0.9011, 0.89, 0.8959, 0.9019, 0.9073, 0.9014, 0.886, 0.8898, 0.8934, 0.8804, 0.903, 0.8959, 0.9003, 0.8961, 0.8932, 0.8827, 0.897, 0.8904, 0.9001, 0.9026, 0.9021, 0.8688, 0.881, 0.9]}

lst = {'aggregation':'trimmed_mean','bias':'0','byz_type':'no','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0839, 0.0901, 0.0949, 0.0972, 0.0984, 0.0998, 0.0997, 0.1023, 0.1039, 0.1043, 0.1066, 0.1081, 0.1102, 0.1137, 0.1166, 0.125, 0.1186, 0.1295, 0.1404, 0.1454, 0.1544, 0.1656, 0.1904, 0.1823, 0.1924, 0.1984, 0.2004, 0.2198, 0.2405, 0.2478, 0.2468, 0.2877, 0.313, 0.3457, 0.3497, 0.3852, 0.4002, 0.4106, 0.4085, 0.4283, 0.4639, 0.4574, 0.4582, 0.4586, 0.4648, 0.4717, 0.4769, 0.5038, 0.509, 0.5184, 0.5255, 0.5291, 0.5301, 0.539, 0.57, 0.5786, 0.5492, 0.5366, 0.5328, 0.5333, 0.5388, 0.5857, 0.6011, 0.6047, 0.6351, 0.6263, 0.6415, 0.6389, 0.6584, 0.6578, 0.6635, 0.6342, 0.6607, 0.6528, 0.6309, 0.7092, 0.7273, 0.7059, 0.7079, 0.6927, 0.745, 0.7336, 0.7003, 0.7105, 0.7314, 0.7724, 0.7556, 0.7335, 0.7591, 0.7654, 0.7803, 0.7205, 0.7302, 0.6355, 0.6165, 0.6658, 0.571, 0.632, 0.6472, 0.6302, 0.6492, 0.6459, 0.7242, 0.6248, 0.6501, 0.6154, 0.7015, 0.702, 0.7437, 0.7375, 0.7973, 0.7604, 0.8255, 0.7858, 0.7976, 0.7219, 0.7024, 0.7245, 0.8029, 0.7765, 0.7922, 0.7391, 0.7024, 0.7154, 0.7489, 0.7411, 0.8372, 0.8399, 0.8429, 0.8345, 0.8433, 0.8382, 0.8568, 0.8542, 0.8303, 0.8324, 0.8408, 0.8186, 0.7879, 0.813, 0.7349, 0.7833, 0.7974, 0.8008, 0.783, 0.8251, 0.8327, 0.8302, 0.8193, 0.842, 0.8631, 0.8613, 0.8017, 0.8378, 0.8454, 0.8364, 0.8321, 0.8193, 0.8061, 0.8353, 0.8285, 0.8205, 0.8593, 0.8564, 0.8089, 0.8418, 0.8218, 0.7781, 0.7833, 0.8177, 0.8277, 0.8451, 0.8273, 0.8084, 0.8299, 0.8396, 0.876, 0.8716, 0.87, 0.873, 0.8709, 0.8771, 0.8874, 0.8821, 0.8764, 0.8543, 0.8468, 0.8398, 0.8832, 0.8833, 0.8651, 0.8759, 0.8436, 0.8248, 0.818, 0.8533, 0.8237, 0.8564, 0.8466, 0.7853, 0.813, 0.8519, 0.8145, 0.7606, 0.8086, 0.8744, 0.8874, 0.8991, 0.8915, 0.8957, 0.8944, 0.8964, 0.8933, 0.8903, 0.8992, 0.8932, 0.8941, 0.8947, 0.8813, 0.8944, 0.8815, 0.8985, 0.8744, 0.8902, 0.8953, 0.8774, 0.897, 0.894, 0.8785, 0.8854, 0.8967, 0.8863, 0.8657, 0.8724, 0.8988, 0.8833, 0.8725, 0.8859, 0.8936, 0.8946, 0.8967, 0.8949, 0.8867, 0.8983, 0.8985, 0.9006, 0.8986, 0.8938, 0.892, 0.8816, 0.8602, 0.8878, 0.9007, 0.902, 0.8989, 0.8872, 0.8508, 0.8213, 0.8465, 0.8704, 0.8916, 0.9062, 0.8989, 0.8995, 0.9059, 0.9062, 0.8974, 0.8774, 0.8985, 0.8999, 0.8995, 0.8924, 0.9056, 0.9092, 0.9089, 0.901, 0.8964, 0.9023, 0.9055, 0.9054, 0.9, 0.9048, 0.9104, 0.9068, 0.8845, 0.9013, 0.91, 0.9127, 0.904, 0.9043, 0.9115, 0.9065, 0.912, 0.9081, 0.9096, 0.8894, 0.9011, 0.9129, 0.9092, 0.9109] }

lst_attack = {'aggregation':'trimmed_mean','bias':'0','byz_type':'trim_attack','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0934, 0.0958, 0.098, 0.1012, 0.1019, 0.1051, 0.1055, 0.1077, 0.1087, 0.1083, 0.1103, 0.1096, 0.1093, 0.1071, 0.1057, 0.1055, 0.1054, 0.1054, 0.1061, 0.1065, 0.1098, 0.1108, 0.1087, 0.1077, 0.1104, 0.1118, 0.1116, 0.1122, 0.1095, 0.1119, 0.108, 0.1076, 0.1078, 0.107, 0.1087, 0.1096, 0.1082, 0.1076, 0.1101, 0.1104, 0.1112, 0.1102, 0.1111, 0.1106, 0.1143, 0.1136, 0.112, 0.1169, 0.1231, 0.1304, 0.1296, 0.1305, 0.1275, 0.1367, 0.1349, 0.1482, 0.1402, 0.1498, 0.1508, 0.1589, 0.1482, 0.1464, 0.1731, 0.1882, 0.1948, 0.2029, 0.1846, 0.199, 0.2187, 0.2275, 0.2203, 0.2824, 0.2922, 0.2709, 0.3042, 0.3291, 0.3485, 0.4075, 0.4232, 0.4222, 0.4028, 0.4119, 0.3894, 0.4435, 0.4808, 0.5009, 0.5339, 0.5404, 0.5251, 0.5515, 0.5678, 0.5719, 0.5878, 0.5722, 0.5442, 0.5587, 0.6067, 0.5835, 0.6432, 0.618, 0.6536, 0.6524, 0.6842, 0.64, 0.6505, 0.631, 0.6574, 0.648, 0.7032, 0.682, 0.7249, 0.7496, 0.7365, 0.7387, 0.7432, 0.7765, 0.7693, 0.7431, 0.7507, 0.7559, 0.7537, 0.7303, 0.6398, 0.6576, 0.7731, 0.7082, 0.7357, 0.7543, 0.7825, 0.7964, 0.7724, 0.7152, 0.7248, 0.7098, 0.6761, 0.7406, 0.7011, 0.6722, 0.6291, 0.6893, 0.6638, 0.715, 0.7936, 0.7904, 0.7986, 0.7713, 0.7771, 0.7599, 0.718, 0.7648, 0.7949, 0.7974, 0.7943, 0.8117, 0.7374, 0.5966, 0.6588, 0.5466, 0.6707, 0.6773, 0.7227, 0.837, 0.8442, 0.842, 0.8293, 0.8389, 0.8535, 0.8563, 0.8311, 0.8592, 0.8491, 0.8383, 0.8191, 0.7881, 0.8359, 0.8446, 0.8647, 0.8655, 0.8227, 0.835, 0.8455, 0.8505, 0.8534, 0.8473, 0.7975, 0.8004, 0.8082, 0.8634, 0.8415, 0.8199, 0.7897, 0.8229, 0.8579, 0.8428, 0.8486, 0.8539, 0.8655, 0.8267, 0.8375, 0.8027, 0.8008, 0.7969, 0.8325, 0.8632, 0.8513, 0.8743, 0.8734, 0.8743, 0.8746, 0.8646, 0.8523, 0.8568, 0.8736, 0.8697, 0.8793, 0.8269, 0.766, 0.7718, 0.7342, 0.7126, 0.7463, 0.7596, 0.7937, 0.8074, 0.8664, 0.8822, 0.8802, 0.8722, 0.862, 0.856, 0.8792, 0.868, 0.8853, 0.8736, 0.8839, 0.886, 0.8771, 0.8494, 0.8578, 0.8799, 0.8782, 0.8819, 0.876, 0.8864, 0.8888, 0.8898, 0.8735, 0.8789, 0.8738, 0.8738, 0.891, 0.8748, 0.8465, 0.8763, 0.8606, 0.878, 0.873, 0.8823, 0.868, 0.8392, 0.8253, 0.8288, 0.8011, 0.8445, 0.8507, 0.8752, 0.8757, 0.8902, 0.8834, 0.8873, 0.8815, 0.8742, 0.887, 0.8843, 0.8799, 0.882, 0.8878, 0.8926, 0.8884, 0.8898, 0.8963, 0.8937, 0.889, 0.8916, 0.8914, 0.8824, 0.8924, 0.8944, 0.8839, 0.8693, 0.8812, 0.8639, 0.8512, 0.8241, 0.827, 0.8782, 0.8935, 0.8949, 0.8751, 0.8971]}

flst = {'aggregation':'fltrust','bias':'0','byz_type':'no','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.106, 0.1055, 0.1139, 0.1208, 0.1314, 0.1303, 0.1703, 0.2283, 0.2246, 0.2356, 0.2566, 0.3023, 0.3345, 0.3558, 0.3425, 0.3068, 0.3601, 0.3591, 0.3961, 0.4336, 0.3944, 0.4817, 0.4701, 0.4957, 0.4974, 0.5102, 0.5479, 0.604, 0.6388, 0.6321, 0.5957, 0.6696, 0.5445, 0.4831, 0.4702, 0.3874, 0.4707, 0.5832, 0.7465, 0.6704, 0.5188, 0.4848, 0.5236, 0.6693, 0.6243, 0.6448, 0.5945, 0.4914, 0.5494, 0.6116, 0.7859, 0.7284, 0.6682, 0.6668, 0.7436, 0.6531, 0.6423, 0.5334, 0.6346, 0.7976, 0.842, 0.8199, 0.7951, 0.7374, 0.6927, 0.6783, 0.7382, 0.7549, 0.8057, 0.791, 0.7119, 0.7902, 0.8162, 0.786, 0.7296, 0.7953, 0.8627, 0.7923, 0.7833, 0.8018, 0.8554, 0.85, 0.8206, 0.8011, 0.821, 0.8392, 0.8754, 0.8357, 0.872, 0.8646, 0.8306, 0.816, 0.7868, 0.8758, 0.8514, 0.8259, 0.8795, 0.8804, 0.879, 0.8873, 0.8825, 0.8881, 0.879, 0.8932, 0.8921, 0.9121, 0.8808, 0.887, 0.8879, 0.8509, 0.8945, 0.9061, 0.9088, 0.8786, 0.877, 0.9108, 0.847, 0.9141, 0.8985, 0.9192, 0.884, 0.8995, 0.8886, 0.9129, 0.9179, 0.9229, 0.8936, 0.8474, 0.9128, 0.9022, 0.9232, 0.9064, 0.9174, 0.9249, 0.9165, 0.9164, 0.9038, 0.9096, 0.9287, 0.9207, 0.9336, 0.9271, 0.9215, 0.9236, 0.9292, 0.9274, 0.9335, 0.9218, 0.9259, 0.9287, 0.9331, 0.9314, 0.9282, 0.928, 0.9333, 0.9307, 0.9231, 0.8881, 0.8713, 0.9362, 0.9305, 0.9405, 0.9243, 0.9388, 0.9341, 0.9329, 0.9392, 0.9319, 0.9364, 0.9302, 0.9488, 0.938, 0.9356, 0.8948, 0.872, 0.9125, 0.9314, 0.924, 0.9326, 0.9354, 0.9327, 0.9429, 0.9352, 0.9413, 0.9303, 0.934, 0.9411, 0.9338, 0.9366, 0.9352, 0.9352, 0.9352, 0.9418, 0.9319, 0.9417, 0.9414, 0.9453, 0.9476, 0.9455, 0.9393, 0.9311, 0.9311, 0.9311, 0.9331, 0.9427, 0.9427, 0.9392, 0.9393, 0.9329, 0.9338, 0.9438, 0.9399, 0.9263, 0.9503, 0.9345, 0.9478, 0.9304, 0.9468, 0.9428, 0.9421, 0.9434, 0.9445, 0.9499, 0.9425, 0.9407, 0.9348, 0.9348, 0.9415, 0.9522, 0.9442, 0.9475, 0.9407, 0.951, 0.9355, 0.9499, 0.9506, 0.9443, 0.9449, 0.9496, 0.9529, 0.9503, 0.9504, 0.948, 0.9495, 0.9433, 0.9417, 0.9481, 0.9509, 0.9368, 0.9448, 0.9463, 0.94, 0.9445, 0.9445, 0.9427, 0.936, 0.9486, 0.9494, 0.9452, 0.943, 0.9361, 0.9476, 0.9468, 0.948, 0.949, 0.949, 0.9499, 0.9455, 0.9455, 0.9507, 0.9469, 0.9482, 0.9439, 0.9489, 0.9492, 0.9494, 0.9493, 0.9504, 0.9517, 0.9393, 0.9475, 0.9511, 0.9492, 0.953, 0.953, 0.9513, 0.9463, 0.9422, 0.9429, 0.941, 0.9449, 0.9447, 0.9431, 0.9399, 0.945, 0.9398, 0.9426, 0.9426, 0.9424, 0.9424]}

flst_attack= {'aggregation':'fltrust','bias':'0','byz_type':'trim_attack','dataset':'MNIST','nbyz':'2','lr':'0.001',
       'key':[0.0934, 0.0958, 0.098, 0.1012, 0.1019, 0.1051, 0.1055, 0.1077, 0.1087, 0.1083, 0.1103, 0.1096, 0.1093, 0.1071, 0.1057, 0.1055, 0.1054, 0.1054, 0.1061, 0.1065, 0.1098, 0.1108, 0.1087, 0.1077, 0.1104, 0.1118, 0.1116, 0.1122, 0.1095, 0.1119, 0.108, 0.1076, 0.1078, 0.107, 0.1087, 0.1096, 0.1082, 0.1076, 0.1101, 0.1104, 0.1112, 0.1102, 0.1111, 0.1106, 0.1143, 0.1136, 0.112, 0.1169, 0.1231, 0.1304, 0.1296, 0.1305, 0.1275, 0.1367, 0.1349, 0.1482, 0.1402, 0.1498, 0.1508, 0.1589, 0.1482, 0.1464, 0.1731, 0.1882, 0.1948, 0.2029, 0.1846, 0.199, 0.2187, 0.2275, 0.2203, 0.2824, 0.2922, 0.2709, 0.3042, 0.3291, 0.3485, 0.4075, 0.4232, 0.4222, 0.4028, 0.4119, 0.3894, 0.4435, 0.4808, 0.5009, 0.5339, 0.5404, 0.5251, 0.5515, 0.5678, 0.5719, 0.5878, 0.5722, 0.5442, 0.5587, 0.6067, 0.5835, 0.6432, 0.618, 0.6536, 0.6524, 0.6842, 0.64, 0.6505, 0.631, 0.6574, 0.648, 0.7032, 0.682, 0.7249, 0.7496, 0.7365, 0.7387, 0.7432, 0.7765, 0.7693, 0.7431, 0.7507, 0.7559, 0.7537, 0.7303, 0.6398, 0.6576, 0.7731, 0.7082, 0.7357, 0.7543, 0.7825, 0.7964, 0.7724, 0.7152, 0.7248, 0.7098, 0.6761, 0.7406, 0.7011, 0.6722, 0.6291, 0.6893, 0.6638, 0.715, 0.7936, 0.7904, 0.7986, 0.7713, 0.7771, 0.7599, 0.718, 0.7648, 0.7949, 0.7974, 0.7943, 0.8117, 0.7374, 0.5966, 0.6588, 0.5466, 0.6707, 0.6773, 0.7227, 0.837, 0.8442, 0.842, 0.8293, 0.8389, 0.8535, 0.8563, 0.8311, 0.8592, 0.8491, 0.8383, 0.8191, 0.7881, 0.8359, 0.8446, 0.8647, 0.8655, 0.8227, 0.835, 0.8455, 0.8505, 0.8534, 0.8473, 0.7975, 0.8004, 0.8082, 0.8634, 0.8415, 0.8199, 0.7897, 0.8229, 0.8579, 0.8428, 0.8486, 0.8539, 0.8655, 0.8267, 0.8375, 0.8027, 0.8008, 0.7969, 0.8325, 0.8632, 0.8513, 0.8743, 0.8734, 0.8743, 0.8746, 0.8646, 0.8523, 0.8568, 0.8736, 0.8697, 0.8793, 0.8269, 0.766, 0.7718, 0.7342, 0.7126, 0.7463, 0.7596, 0.7937, 0.8074, 0.8664, 0.8822, 0.8802, 0.8722, 0.862, 0.856, 0.8792, 0.868, 0.8853, 0.8736, 0.8839, 0.886, 0.8771, 0.8494, 0.8578, 0.8799, 0.8782, 0.8819, 0.876, 0.8864, 0.8888, 0.8898, 0.8735, 0.8789, 0.8738, 0.8738, 0.891, 0.8748, 0.8465, 0.8763, 0.8606, 0.878, 0.873, 0.8823, 0.868, 0.8392, 0.8253, 0.8288, 0.8011, 0.8445, 0.8507, 0.8752, 0.8757, 0.8902, 0.8834, 0.8873, 0.8815, 0.8742, 0.887, 0.8843, 0.8799, 0.882, 0.8878, 0.8926, 0.8884, 0.8898, 0.8963, 0.8937, 0.889, 0.8916, 0.8914, 0.8824, 0.8924, 0.8944, 0.8839, 0.8693, 0.8812, 0.8639, 0.8512, 0.8241, 0.827, 0.8782, 0.8935, 0.8949, 0.8751, 0.8971]}




# 第二个子图是 Accuracy
plt.plot(range(len(krum["key"])), krum["key"], label='krum')
plt.plot(range(len(krum_attack["key"])), krum_attack["key"], label='krum_attack')
#plt.title(f'Test Accuracy Over Iterations\nAggregation: {lst["aggregation"]}, Non-iid: {lst["bias"]}, Attack Type: {lst["byz_type"]}\nDataset: {lst["dataset"]},Attack Number:{lst["nbyz"]},LR:{lst["lr"]}')
plt.title(f'Test Accuracy Over Iterations\nAggregation: {krum["aggregation"]}, Attack Type: {krum_attack["byz_type"]}, Dataset: {krum["dataset"]}')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
# 显示图例
plt.legend()

# 显示图
plt.show()


