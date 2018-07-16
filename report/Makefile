# Note: Tabs must be used instead of spaces
#       Using spaces results in a "missing separator" error
filename = ikr-protocol-comparison
pdf: clean
	pdflatex $(filename)    # Creates a bad pdf and lots of temp files
	bibtex $(filename)      # Creates filename.bbl, needed by latex
	pdflatex $(filename)    # Produces a slightly better pdf and temp files
	pdflatex $(filename)    # Produces a correct pdf from the temp files
	make clean
clean:
	rm -f $(filename).aux $(filename).log $(filename).out $(filename).spl
	rm -f $(filename).blg $(filename).bbl $(filename).toc
	rm -f $(filename).maf $(filename).mtc*
	rm -f *.aux
