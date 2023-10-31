#pragma once


#include <string>
#include <vector>

struct I2 {
	int x;
	int y;
};

class GroundTruth {
public:
	explicit GroundTruth(const std::string& source);

	const std::vector<I2>& getYellow() const { return yellow; }
	const std::vector<I2>& getBlue() const { return blue; }
	const std::vector<I2>& getOrange() const { return orange; }
	const std::vector<I2>& getGreen() const { return green; }
	const std::vector<I2>& getPink() const { return pink; }

private:
	std::vector<I2> yellow;
	std::vector<I2> blue;
	std::vector<I2> orange;
	std::vector<I2> green;
	std::vector<I2> pink;
};
