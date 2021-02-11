CC = gcc
IDIR = include
SDIR = src
ODIR = build

CFLAGS = -O2 -I $(IDIR)
AVXFLAGS = -march=skylake-avx512
# AVXFLAGS = -mavx2
DEPS = $(wildcard $(IDIR/)*.h)
ALLMAINS := $(ODIR)/cache_aware_matmul.o $(ODIR)/basic_matmul_col_major.o  $(ODIR)/cache_oblivious_matmul.o $(ODIR)/basic_matmul.o $(ODIR)/basic_matmul_avx.o $(ODIR)/cache_aware_matmul_avx.o $(ODIR)/cache_aware_matmul_col_major.o $(ODIR)/cache_oblivious_matmul_col_major.o $(ODIR)/basic_matmul_avx_unroll.o $(ODIR)/cache_aware_matmul_avx_unroll.o $(ODIR)/cache_oblivious_matmul_avx.o

BASICMATMUL = basicmatmul
BASICMATMUL_IGNORE = $(filter-out $(ODIR)/basic_matmul.o, $(ALLMAINS))

CACHEAWAREMATMUL = cacheaware 
CACHEAWAREMATMUL_IGNORE = $(filter-out $(ODIR)/cache_aware_matmul.o, $(ALLMAINS))

CACHEOBVMATMUL = cacheoblivious
CACHEOBVMATMUL_IGNORE = $(filter-out $(ODIR)/cache_oblivious_matmul.o, $(ALLMAINS))

CACHEOBVMATMUL_AVX = cacheoblivious_avx
CACHEOBVMATMUL_AVX_IGNORE = $(filter-out $(ODIR)/cache_oblivious_matmul_avx.o, $(ALLMAINS))

BASICMATMUL_AVX = basicmatmul_avx
BASICMATMUL_AVX_IGNORE = $(filter-out $(ODIR)/basic_matmul_avx.o, $(ALLMAINS))

BASICMATMUL_AVX_UNROLL = basicmatmul_avx_unroll
BASICMATMUL_AVX_UNROLL_IGNORE = $(filter-out $(ODIR)/basic_matmul_avx_unroll.o, $(ALLMAINS))

CACHEAWAREMATMUL_AVX = cacheaware_avx
CACHEAWAREMATMUL_AVX_IGNORE = $(filter-out $(ODIR)/cache_aware_matmul_avx.o, $(ALLMAINS))

CACHEAWAREMATMUL_AVX_UNROLL = cacheaware_avx_unroll
CACHEAWAREMATMUL_AVX_UNROLL_IGNORE = $(filter-out $(ODIR)/cache_aware_matmul_avx_unroll.o, $(ALLMAINS))

BASICMATMUL_COL_MAJOR = basicmatmul_col_major
BASICMATMUL_COL_MAJOR_IGNORE = $(filter-out $(ODIR)/basic_matmul_col_major.o, $(ALLMAINS))

CACHEAWAREMATMUL_COL_MAJOR = cacheaware_col_major
CACHEAWAREMATMUL_COL_MAJOR_IGNORE = $(filter-out $(ODIR)/cache_aware_matmul_col_major.o, $(ALLMAINS))

SRCS := $(wildcard $(SDIR)/*.c)
OBJS := $(patsubst $(SDIR)/%.c, $(ODIR)/%.o,$(SRCS))

all: $(BASICMATMUL) $(CACHEAWAREMATMUL) $(CACHEOBVMATMUL) $(BASICMATMUL_AVX) $(CACHEAWAREMATMUL_AVX) $(BASICMATMUL_COL_MAJOR) $(CACHEAWAREMATMUL_COL_MAJOR) $(BASICMATMUL_AVX_UNROLL) $(CACHEAWAREMATMUL_AVX_UNROLL) $(CACHEOBVMATMUL_AVX)

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS) $(AVXFLAGS)

$(BASICMATMUL): $(filter-out $(BASICMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(BASICMATMUL_AVX): $(filter-out $(BASICMATMUL_AVX_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) $(AVXFLAGS)

$(BASICMATMUL_AVX_UNROLL): $(filter-out $(BASICMATMUL_AVX_UNROLL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) $(AVXFLAGS)

$(CACHEAWAREMATMUL): $(filter-out $(CACHEAWAREMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) 

$(CACHEAWAREMATMUL_AVX): $(filter-out $(CACHEAWAREMATMUL_AVX_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) 

$(CACHEAWAREMATMUL_AVX_UNROLL): $(filter-out $(CACHEAWAREMATMUL_AVX_UNROLL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS) 

$(CACHEOBVMATMUL): $(filter-out $(CACHEOBVMATMUL_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(CACHEOBVMATMUL_AVX): $(filter-out $(CACHEOBVMATMUL_AVX_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(BASICMATMUL_COL_MAJOR): $(filter-out $(BASICMATMUL_COL_MAJOR_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

$(CACHEAWAREMATMUL_COL_MAJOR): $(filter-out $(CACHEAWAREMATMUL_COL_MAJOR_IGNORE), $(OBJS))
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf $(ODIR) $(CACHEAWAREMATMUL) $(BASICMATMUL_COL_MAJOR) $(BASICMATMUL) $(CACHEAWAREMATMUL) $(CACHEOBVMATMUL) $(BASICMATMUL_AVX) $(CACHEAWAREMATMUL_COL_MAJOR) $(BASICMATMUL_AVX_UNROLL) $(CACHEAWAREMATMUL_AVX_UNROLL) $(CACHEAWAREMATMUL_AVX_UNROLL) $(CACHEOBVMATMUL_AVX)




