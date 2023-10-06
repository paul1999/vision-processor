#pragma once


#include <vector>
#include <memory>

enum PixelFormat {
	RGGB8,
	BGR888,
	U8,
	I8,
	F32,

	NV12
};


class Image {
public:
	Image(PixelFormat format, int width, int height, unsigned char* data): format(format), width(width), height(height), data(data) {}
	virtual ~Image() = default;

	[[nodiscard]] PixelFormat getFormat() const { return format; }
	[[nodiscard]] int getWidth() const { return width; }
	[[nodiscard]] int getHeight() const { return height; }
	[[nodiscard]] unsigned char* getData() const { return data; }

	int pixelWidth();
	int pixelHeight();
	int pixelSize();

private:
	const PixelFormat format;
	const int width;
	const int height;
	unsigned char* data;
};

class BufferImage : public Image {
public:
	static std::shared_ptr<Image> create(PixelFormat format, int width, int height);

	BufferImage(PixelFormat format, int width, int height, unsigned char* data);
	~BufferImage() override;
};