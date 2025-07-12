#
# TransFig makefile
#

all: gameMINMAXsigned.eps 

# translation into eps

gameMINMAXsigned.tex: gameMINMAXsigned.eps Makefile
	fig2ps2tex gameMINMAXsigned.eps >gameMINMAXsigned.tex
clean::
	rm -f gameMINMAXsigned.eps

gameMINMAXsigned.eps: gameMINMAXsigned.fig Makefile
	fig2dev -L eps gameMINMAXsigned.fig gameMINMAXsigned.eps
clean::
	rm -f gameMINMAXsigned.eps

