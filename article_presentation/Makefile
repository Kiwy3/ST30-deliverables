all: main.pdf

LATEXCMD=lualatex

%.pdf: %.tex
	TEXINPUTS=.:theme//:latex-inria-fonts//: \
	${LATEXCMD} -file-line-error -halt-on-error $<

clean:
	@rm -f main.aux main.log main.nav main.out main.snm main.toc main.pdf main.fls main.fdb_latexmk
