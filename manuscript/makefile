help:
	@echo "Available makes:"
	@echo "figs - rename, convert to tiff and set colorpace, dpi and size of figures"
	@echo "clean - remove unused files"

all:
	make figs
	make clean

# Convert and rename figures
figs:
	convert Fig/Figure1.png -colorspace CMYK -units PixelsPerInch Fig/Figure1.tiff
	convert Fig/Figure2.png -colorspace CMYK -units PixelsPerInch Fig/Figure2.tiff
	convert Fig/Figure3.png -colorspace CMYK -units PixelsPerInch Fig/Figure3.tiff
	convert Fig/Figure4.png -colorspace CMYK -units PixelsPerInch Fig/Figure4.tiff
	convert Fig/Figure5.png -colorspace CMYK -units PixelsPerInch Fig/Figure5.tiff
	convert Fig/Figure6.png -colorspace CMYK -units PixelsPerInch Fig/Figure6.tiff
	convert Fig/Figure7.png -colorspace CMYK -units PixelsPerInch Fig/Figure7.tiff
	convert Fig/Figure8.png -colorspace CMYK -units PixelsPerInch Fig/Figure8.tiff
	convert Fig/Figure9.png -colorspace CMYK -units PixelsPerInch Fig/Figure9.tiff
	convert Fig/Figure10.png -colorspace CMYK -units PixelsPerInch Fig/Figure10.tiff
	convert Fig/Figure11.png -colorspace CMYK -units PixelsPerInch Fig/Figure11.tiff
	

# Remove unused files
clean:
	rm *.aux *.bbl *.blg *.log *.pdf *.lof *.lot *.synctex.gz *.out

