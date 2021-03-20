CC = gcc
GLC = glslc

CFLAGS = -Wall -Wno-missing-braces -Wno-attributes -fPIC
LDFLAGS = -L/opt/hfs18.0/dsolib -L$(HOME)/lib
INFLAGS = -I$(HOME)/dev
GLFLAGS = --target-env=vulkan1.2
LIBS = -lobsidian -lvulkan -lcoal
LIBNAME = tanto
DEV = $(HOME)/dev

O = build
GLSL = shaders
SPV  = shaders/spv

DEPS =  \

SHDEPS = \
		shaders/common.glsl \
		shaders/vert-common.glsl \
		shaders/frag-common.glsl

debug: CFLAGS += -g -DVERBOSE=1
debug: all

release: CFLAGS += -DNDEBUG -O3
release: all

all: coal obsidian tanto tags shaders

FRAG  := $(patsubst %.frag,$(SPV)/%-frag.spv,$(notdir $(wildcard $(GLSL)/*.frag)))
VERT  := $(patsubst %.vert,$(SPV)/%-vert.spv,$(notdir $(wildcard $(GLSL)/*.vert)))
RGEN  := $(patsubst %.rgen,$(SPV)/%-rgen.spv,$(notdir $(wildcard $(GLSL)/*.rgen)))
RCHIT := $(patsubst %.rchit,$(SPV)/%-rchit.spv,$(notdir $(wildcard $(GLSL)/*.rchit)))
RMISS := $(patsubst %.rmiss,$(SPV)/%-rmiss.spv,$(notdir $(wildcard $(GLSL)/*.rmiss)))

shaders: $(FRAG) $(VERT) $(RGEN) $(RCHIT) $(RMISS)

clean: 
	rm -f $(O)/*.o $(LIBNAME).so $(LIB)/$(LIBNAME) $(SPV)/* 

tags:
	ctags -R .

.PHONY: obsidian
obsidian:
	make -C $(DEV)/obsidian

.PHONY: coal
coal:
	make -C $(DEV)/coal

tanto: $(O)/render.o 
	$(CC) $(LDFLAGS) -shared -o tanto.so $< $(LIBS)

bin: main.c $(OBJS) $(DEPS) shaders
	$(CC) $(CFLAGS) $(INFLAGS) $(LDFLAGS) $(OBJS) $< -o $(BIN)/$(NAME) $(LIBS)

lib: $(OBJS) $(DEPS) shaders
	$(CC) -shared -o $(LIB)/lib$(LIBNAME).so $(OBJS) 

$(O)/%.o:  %.c $(DEPS)
	$(CC) $(CFLAGS) $(INFLAGS) -c $< -o $@

$(SPV)/%-vert.spv: $(GLSL)/%.vert $(SHDEPS)
	$(GLC) $(GLFLAGS) $< -o $@

$(SPV)/%-frag.spv: $(GLSL)/%.frag $(SHDEPS)
	$(GLC) $(GLFLAGS) $< -o $@

$(SPV)/%-rchit.spv: $(GLSL)/%.rchit $(SHDEPS)
	$(GLC) $(GLFLAGS) $< -o $@

$(SPV)/%-rgen.spv: $(GLSL)/%.rgen $(SHDEPS)
	$(GLC) $(GLFLAGS) $< -o $@

$(SPV)/%-rmiss.spv: $(GLSL)/%.rmiss $(SHDEPS)
	$(GLC) $(GLFLAGS) $< -o $@
