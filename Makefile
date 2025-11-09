CC      := gcc
CFLAGS  := -ansi -Wall -Wextra -Werror -pedantic-errors -O2
LDLIBS  := -lm

all: symnmf

symnmf: symnmf_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

symnmf_cli.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -DSYMNMF_CLI -c $< -o $@

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c $< -o $@

ext:
	python3 setup.py build_ext --inplace

clean:
	rm -f symnmf *.o
	rm -rf build *.so __pycache__
