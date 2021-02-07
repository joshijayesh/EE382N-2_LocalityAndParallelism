CC = gcc
IDIR = include
SDIR = src
ODIR = build

CFLAGS = -O2 -I $(IDIR)
DEPS = $(wildcard $(IDIR/)*.h)

BASICMATMUL = basicmatmul
BASICMATMUL_IGNORE = $(ODIR)/cache_aware_matmul.o $(ODIR)/cache_oblivious_matmul.o

CACHEAWAREMATMUL = cacheaware 
CACHEAWAREMATMUL_IGNORE = $(ODIR)/basic_matmul.o $(ODIR)/cache_oblivious_matmul.o

CACHEOBVMATMUL = cacheoblivious
CACHEOBVMATMUL_IGNORE = $(ODIR)/basic_matmul.o $(ODIR)/cache_aware_matmul.o

SRCS := $(wildcard $(SDIR)/*.c)
OBJS := $(patsubst $(SDIR)/%.c, $(ODIR)/%.o,$(SRCS))

all: $(BASICMATMUL) $(CACHEAWAREMATMUL) $(CACHEOBVMATMUL)

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BASICMATMUL): $(filter-out $(BASICMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(CACHEAWAREMATMUL): $(filter-out $(CACHEAWAREMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(CACHEOBVMATMUL): $(filter-out $(CACHEOBVMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf $(ODIR) $(BASICMATMUL) $(CACHEAWAREMATMUL) $(CACHEOBVMATMUL)

