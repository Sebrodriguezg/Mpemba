include ../../this_dir.mk
include ../../options.mk

#Define Flags ----------

TENSOR_HEADERS=$(PREFIX)/itensor/all.h
CCFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS)
CCGFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS)
LIBFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBFLAGS)
LIBGFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBGFLAGS)

#Rules ------------------

%.o: %.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: mpembaq1 opspreadingq1 mpembaq2 opspreadingq2

debug: mpembaq1 opspreadingq1 mpembaq2 opspreadingq2

all: mpembaq1 opspreadingq1 mpembaq2 opspreadingq2

mpembaq1: mpembaq1.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) mpembaq1.o -o mpembaq1 $(LIBFLAGS)

opspreadingq1: opspreadingq1.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) opspreadingq1.o -o opspreadingq1 $(LIBFLAGS)

mpembaq2: mpembaq2.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) mpembaq2.o -o mpembaq2 $(LIBFLAGS)

opspreadingq2: opspreadingq2.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) opspreadingq2.o -o opspreadingq2 $(LIBFLAGS)

