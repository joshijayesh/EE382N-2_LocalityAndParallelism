CC = gcc
IDIR = include
SDIR = src
ODIR = build

CFLAGS = -O2 -I $(IDIR)
AVXFLAGS = -march=skylake-avx512
# AVXFLAGS = -mavx2
DEPS = $(wildcard $(IDIR/)*.h)
ALLMAINS := $(ODIR)/cache_aware_matmul.o $(ODIR)/cache_oblivious_matmul.o $(ODIR)/basic_matmul.o $(ODIR)/basic_matmul_avx.o $(ODIR)/cache_aware_matmul_avx.o

BASICMATMUL = basicmatmul
BASICMATMUL_IGNORE = $(filter-out $(ODIR)/basic_matmul.o, $(ALLMAINS))

CACHEAWAREMATMUL = cacheaware 
CACHEAWAREMATMUL_IGNORE = $(filter-out $(ODIR)/cache_aware_matmul.o, $(ALLMAINS))

CACHEOBVMATMUL = cacheoblivious
CACHEOBVMATMUL_IGNORE = $(filter-out $(ODIR)/cache_oblivious_matmul.o, $(ALLMAINS))

BASICMATMUL_AVX = basicmatmul_avx
BASICMATMUL_AVX_IGNORE = $(filter-out $(ODIR)/basic_matmul_avx.o, $(ALLMAINS))

CACHEAWAREMATMUL_AVX = cacheaware_avx
CACHEAWAREMATMUL_AVX_IGNORE = $(filter-out $(ODIR)/cache_aware_matmul_avx.o, $(ALLMAINS))

SRCS := $(wildcard $(SDIR)/*.c)
OBJS := $(patsubst $(SDIR)/%.c, $(ODIR)/%.o,$(SRCS))

all: $(BASICMATMUL) $(CACHEAWAREMATMUL) $(CACHEOBVMATMUL) $(BASICMATMUL_AVX) $(CACHEAWAREMATMUL_AVX)

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS) $(AVXFLAGS)

$(BASICMATMUL): $(filter-out $(BASICMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(BASICMATMUL_AVX): $(filter-out $(BASICMATMUL_AVX_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) $(AVXFLAGS)

$(CACHEAWAREMATMUL): $(filter-out $(CACHEAWAREMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) 

$(CACHEAWAREMATMUL_AVX): $(filter-out $(CACHEAWAREMATMUL_AVX_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) 

$(CACHEOBVMATMUL): $(filter-out $(CACHEOBVMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf $(ODIR) $(BASICMATMUL) $(CACHEAWAREMATMUL) $(CACHEOBVMATMUL) $(BASICMATMUL_AVX)

