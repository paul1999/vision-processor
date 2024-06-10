#pragma once

//1 indicates green, 0 pink, increasing 2d angle starting from bot orientation most significant bit to least significant bit
const int patterns[16] = {
		0b0100, // 0
		0b1100, // 1
		0b1101, // 2
		0b0101, // 3
		0b0010, // 4
		0b1010, // 5
		0b1011, // 6
		0b0011, // 7
		0b1111, // 8
		0b0000, // 9
		0b0110, //10
		0b1001, //11
		0b1110, //12
		0b1000, //13
		0b0111, //14
		0b0001  //15
};
const int patternLUT[16] = { 9, 15, 4, 7, 0, 3, 10, 14, 13, 11, 5, 6, 1, 2, 12, 8 };

const double patternAngles[4] = {
		1.0021839078803572,
		2.5729802346752537,
		-2.5729802346752537, //3.7102050725043325
		-1.0021839078803572 //5.281001399299229
};