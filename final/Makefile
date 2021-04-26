CC = g++
IDIR = include cxxopts/include
SRC_DIR = src
ODIR = build
LIBS = 

SRCS := $(shell find $(SRC_DIR)/ -name '*.cpp')
OBJS := $(addprefix $(ODIR)/, $(notdir $(SRCS:%.cpp=%.o)))

MAIN = eigenfaces
CCFLAGS = $(addprefix -I , $(IDIR))


all: $(MAIN) 

$(ODIR)/%.o:
	mkdir -p $(ODIR)
	$(CC) -c $(filter %$(notdir $(basename $@)).cpp, $(SRCS)) -o $@ $< $(CCFLAGS) $(LIBS)

$(MAIN): $(OBJS)
	$(CC) -o $@ $^ $(CCFLAGS) $(LIBS)

#$(CC) -o $@ $^ $(CCFLAGS)

.PHONE: clean

clean:
	rm -rf $(ODIR) $(MAIN)

