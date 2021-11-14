# codes_centres[index] = [place_no, angle, x, y]
# code = place_no
codes_centres = [[0, 90, 323, 329], # right edge of paper is 70 cm from exit of big room, 533 - 46 -77.5 - 70 - 21 / 2 = 329
                 [1, 90, 323, 209], # right edge of paper is 190 cm from exit of big room, 533 - 46 -77.5 - 190 - 21 / 2 = 209
                 [2, -90, 0, 189], # center of paper is in 189 cm from balcony wall
                 [3, -90, 0, 289], # center of paper is in 289 cm from balcony wall
                 [4, 0,  174, 80], # center of paper is in 80 cm from balcony wall and 149 cm from balcony right wall, 323 - 149 = 174
                 [5, 180, 204.5, 313.5], # center of paper is 96 cm from exit of big room in y direction, and in 118.5 from balcony right wall in -x direction, 323 - 118.5 = 204.5, 533 - 46 -77.5 - 96 = 313.5
                 [6, 180, 238, 533], # center of paper is in 85 cm from right balcony whall  in -x direction, on the wardrobe wall, 323-85 = 238
                 [7, 180, 334, 487], # on left edge of exit of big room, 323 + 22 / 2 = 334, 533 - 46  = 487
                 [8, 180, 204.5, 403.5], # like 5 but shiftet at 90 cm to +y direction
                 [9, 148, 288.5, 504], # like 7 but with shift (-45.5, +17) cm, 32 degrees, 180 - 32 = 148
                 [10, 90, 424.8, 404.5], #  shift (+79.8, -5) from far (+x) enge of rigth edge of exit of big room 323 + 22 + 79.8 = 424.8, 533 - 46 - 77.5 - 5 = 404.5
                 [11, 0, 474.3, 417.1], # shift (+49.5, +12.6) from code 10 center
                 [12, 180, 502, 566], # shift of (+157, 0) from (+y, -x)-corner (ignoring column) of corridor 323 + 22 + 157 = 502, 533 - 46 + 79 = 566
                 [13, 0, 630.3, 354.5], # shift to -x at 39.7 from internal corner in coridor when corridor turn to kitchen, 323 + 22 + 268 + 55.5 + 102 - 100.5 - 39.7 = 630.3, 533 - 46 - 77.5 - 55 = 354.5
                 [14, 90, 770.5, 535.5], # in (+x, +y) - corner of corridor, 323 + 22 + 268 + 55.5 + 102 = 770.5, 533 - 46 + 79 - 36 + 16 - 21 / 2 = 535.5
                 [15, 90, 770.5, 426], # shift (0, -120) from (+x, +y) - corner of corridor, 323 + 22 + 268 + 55.5 + 102 = 770.5, 533 - 46 + 79 - 36 + 16 - 120 = 426
                 [16, 90, 562.5, 503.8], # like 12 but shifted at (+60.5, -62.2)
                 [17, 0, 760.0, 223.5], # left corner at enterance to citchen, 323 + 22 + 268 + 55.5 + 102     - 10.5 = 760.0, 533 - 46 + 79 - 36 + 16      - 322.5 = 223.5
                 #[18, -90, 670.0, 284.5], # shift at (0, -70) right corner of enterence of corridor to citchen, y-ancor from 11 323 + 22 + 268 + 55.5 + 102  - 100.5 = 670.0    354.5 - 70 = 284.5
                 [18, -90, 670.0, 234.5], # shift at (0, -50) relative to old position
                 [19, 0, 700.0, 4.5], # like 17 but -219 by y, and like 18 but +30 cm by x, 670.0 + 30 = 700.0   223.5 - 219 = 4.5
                 [20, 0, 750.0, 4.5],  # like 17 but -219 by y, and like 18 but +80 cm by x, 670.0 + 30 = 750.0   223.5 - 219 = 4.5
                 #[21, 90, 922.0, 99], # ancor is corner of scafs: like 15 for x, like 17 for y but with shift (770.5 + 19, 223.5 - 19 - 28) = (789.5, 176.5), shift to corner (+150, -60), in total corener: (939.5, 116.5) and shift at -17.5 by y
                 [21, 90, 934.0, 93.5], # like 20 but (+184, +89)
                 [22, 90, 934.0, 31.5], # like 22 but (0, -62)
                 [23, 90, 770.5, 321.5], # shift at (+10.5, +98) relative to 17
                 [24, -90, 670.0, 294.5]] # (0, +10) relateve to old 18