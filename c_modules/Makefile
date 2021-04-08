TARGETS=test

all: $(TARGETS)

LIBS=../c_modules/utils.c
DIRS=-I../c_modules -L../c_modules
FLAGS=-std=gnu99 -Wall -g #-Werror

clean:
	for f in $(TARGETS); do \
		if [ -f $$f ]; then rm $$f; fi; \
		if [ -f $$f.exe ]; then rm $$f.exe; fi; \
	done

rebuild: clean all

test: test.c
	gcc $(FLAGS) $@.c -o $@ $(DIRS) $(LIBS) -lm