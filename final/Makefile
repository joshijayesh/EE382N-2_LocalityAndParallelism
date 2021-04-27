CC = g++
IDIR = include cxxopts/include
CUIDIR = include
SRC_DIR = src
ODIR = build
CUODIR = $(ODIR)/cu
LIBS = 
CUDA_PATH=$(TACC_CUDA_DIR)

SRCS := $(shell find $(SRC_DIR)/ -name '*.cpp')
CUSRCS := $(shell find $(SRC_DIR)/ -name '*.cu')
OBJS := $(addprefix $(ODIR)/, $(notdir $(SRCS:%.cpp=%.o)))
CUOBJS := $(addprefix $(CUODIR)/, $(notdir $(CUSRCS:%.cu=%.o)))

MAIN = eigenfaces
INCLUDES = $(addprefix -I , $(IDIR))
CCFLAGS = -O3 -Wall $(INCLUDES)


# Select target arch
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   := $(GENCODE_SM60) # For Pascal architecture

NVCC_LDFLAGS = -L$(CUDA_PATH)/lib64/ -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin=$(CC)
NVCC_INCLUDES = $(addprefix -I , $(CUIDIR))
NVCC_FLAGS = -O3 -m64 $(GENCODE_FLAGS) $(NVCC_INCLUDES)



all: $(MAIN) 

$(ODIR)/%.o:
	mkdir -p $(ODIR)
	$(CC) -c $(filter %$(notdir $(basename $@)).cpp, $(SRCS)) -o $@ $< $(CCFLAGS) $(LIBS)

$(CUODIR)/%.o:
	mkdir -p $(CUODIR)
	$(NVCC) $(filter %$(notdir $(basename $@)).cu, $(CUSRCS)) -c -o $@ $< $(NVCC_FLAGS)

$(MAIN): $(OBJS) $(CUOBJS)
	$(CC) -o $@ $^ $(CCFLAGS) $(LIBS) $(NVCC_LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(ODIR) $(MAIN)

