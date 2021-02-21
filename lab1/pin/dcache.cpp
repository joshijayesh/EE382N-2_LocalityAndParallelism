/*
 * Copyright 2002-2020 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */

/*! @file
 *  This file contains an ISA-portable cache simulator
 *  data cache hierarchies
 */


#include "pin.H"

#include <iostream>
#include <fstream>

#include "dcache.H"
#include "pin_profile.H"
using std::ostringstream;
using std::string;
using std::cerr;
using std::endl;


std::ofstream outFile;

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE,    "pintool",
    "o", "dcache.out", "specify dcache file name");
KNOB<BOOL>   KnobTrackLoads(KNOB_MODE_WRITEONCE,    "pintool",
    "tl", "0", "track individual loads -- increases profiling time");
KNOB<BOOL>   KnobTrackStores(KNOB_MODE_WRITEONCE,   "pintool",
   "ts", "0", "track individual stores -- increases profiling time");
KNOB<UINT32> KnobThresholdHit(KNOB_MODE_WRITEONCE , "pintool",
   "rh", "100", "only report memops with hit count above threshold");
KNOB<UINT32> KnobThresholdMiss(KNOB_MODE_WRITEONCE, "pintool",
   "rm","100", "only report memops with miss count above threshold");
KNOB<UINT32> KnobL1CacheSize(KNOB_MODE_WRITEONCE, "pintool",
    "c1","32", "cache size in kilobytes");
KNOB<UINT32> KnobL1LineSize(KNOB_MODE_WRITEONCE, "pintool",
    "b1","32", "cache block size in bytes");
KNOB<UINT32> KnobL1Associativity(KNOB_MODE_WRITEONCE, "pintool",
    "a1","4", "cache associativity (1 for direct mapped)");
KNOB<UINT32> KnobL2CacheSize(KNOB_MODE_WRITEONCE, "pintool",
    "c2","2048", "cache size in kilobytes");
KNOB<UINT32> KnobL2LineSize(KNOB_MODE_WRITEONCE, "pintool",
    "b2","64", "cache block size in bytes");
KNOB<UINT32> KnobL2Associativity(KNOB_MODE_WRITEONCE, "pintool",
    "a2","16", "cache associativity (1 for direct mapped)");

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage()
{
    cerr <<
        "This tool represents a cache simulator.\n"
        "\n";

    cerr << KNOB_BASE::StringKnobSummary() << endl; 
    return -1;
}

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */

// wrap configuation constants into their own name space to avoid name clashes
namespace DL1
{
    const UINT32 max_sets = KILO; // cacheSize / (lineSize * associativity);
    const UINT32 max_associativity = 256; // associativity;
    const CACHE_ALLOC::STORE_ALLOCATION allocation = CACHE_ALLOC::STORE_ALLOCATE;

    typedef CACHE_LRU(max_sets, max_associativity, allocation) CACHE;
}

DL1::CACHE* dl1 = NULL;

namespace DL2
{
    const UINT32 max_sets = 32 * KILO;  // cacheSize / (lineSize * associativity);
    const UINT32 max_associativity = 256; // associativity;
    const CACHE_ALLOC::STORE_ALLOCATION allocation = CACHE_ALLOC::STORE_ALLOCATE;

    typedef CACHE_LRU(max_sets, max_associativity, allocation) CACHE;
}

DL2::CACHE* dl2 = NULL;

typedef enum
{
    COUNTER_MISS = 0,
    COUNTER_HIT = 1,
    COUNTER_NUM
} COUNTER;

typedef  COUNTER_ARRAY<UINT64, COUNTER_NUM> COUNTER_HIT_MISS;


// holds the counters with misses and hits
// conceptually this is an array indexed by instruction address
COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> profile_l1;
COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> profile_l2;

/* ===================================================================== */

VOID LoadMulti(ADDRINT addr, UINT32 size, UINT32 instId, UINT32 instId2)
{
    // first level D-cache
    const AccessResult dl1Result = dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_LOAD);

    const COUNTER counter = dl1Result.hit ? COUNTER_HIT : COUNTER_MISS;
    profile_l1[instId][counter]++;

    if(dl1Result.hit) {
        // second level D-cache
        const AccessResult dl2Result = dl2->Access(addr, size, CACHE_BASE::ACCESS_TYPE_LOAD);

        const COUNTER counter2 = dl2Result.hit ? COUNTER_HIT : COUNTER_MISS;
        profile_l2[instId2][counter2]++;
    }
}

/* ===================================================================== */

VOID StoreMulti(ADDRINT addr, UINT32 size, UINT32 instId, UINT32 instId2)
{
    // first level D-cache
    const AccessResult dl1Result = dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_STORE);

    const COUNTER counter = dl1Result.hit ? COUNTER_HIT : COUNTER_MISS;
    profile_l1[instId][counter]++;

    if(dl1Result.hit) {
        // second level D-cache
        const AccessResult dl2Result = dl2->Access(addr, size, CACHE_BASE::ACCESS_TYPE_STORE);

        const COUNTER counter2 = dl2Result.hit ? COUNTER_HIT : COUNTER_MISS;
        profile_l2[instId2][counter2]++;
    }
}

/* ===================================================================== */

VOID LoadSingle(ADDRINT addr, UINT32 instId, UINT32 instId2)
{
    // @todo we may access several cache lines for 
    // first level D-cache
    const AccessResult dl1Result = dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_LOAD);

    const COUNTER counter = dl1Result.hit ? COUNTER_HIT : COUNTER_MISS;
    profile_l1[instId][counter]++;

    if(dl1Result.hit) {
        const AccessResult dl2Result = dl2->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_LOAD);

        const COUNTER counter2 = dl2Result.hit ? COUNTER_HIT : COUNTER_MISS;
        profile_l2[instId2][counter2]++;
    }
}
/* ===================================================================== */

VOID StoreSingle(ADDRINT addr, UINT32 instId, UINT32 instId2)
{
    // @todo we may access several cache lines for 
    // first level D-cache
    const AccessResult dl1Result = dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_STORE);

    const COUNTER counter = dl1Result.hit ? COUNTER_HIT : COUNTER_MISS;
    profile_l1[instId][counter]++;

    if(dl1Result.hit) {
        const AccessResult dl2Result = dl2->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_STORE);

        const COUNTER counter2 = dl2Result.hit ? COUNTER_HIT : COUNTER_MISS;
        profile_l2[instId2][counter2]++;
    }
}

/* ===================================================================== */

VOID LoadMultiFast(ADDRINT addr, UINT32 size)
{
    const AccessResult dl1Result = dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_LOAD);
    if(!dl1Result.hit) dl2->Access(addr, size, CACHE_BASE::ACCESS_TYPE_LOAD);
    if(dl1Result.dirty_evicted) dl2->WriteBack(CACHE_BASE::ACCESS_TYPE_LOAD);
}

/* ===================================================================== */

VOID StoreMultiFast(ADDRINT addr, UINT32 size)
{
    const AccessResult dl1Result = dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_STORE);
    if(!dl1Result.hit) dl2->Access(addr, size, CACHE_BASE::ACCESS_TYPE_STORE);
    if(dl1Result.dirty_evicted) dl2->WriteBack(CACHE_BASE::ACCESS_TYPE_STORE);
}

/* ===================================================================== */

VOID LoadSingleFast(ADDRINT addr)
{
    const AccessResult dl1Result = dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_LOAD);    
    if(!dl1Result.hit) dl2->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_LOAD);    
    if(dl1Result.dirty_evicted) dl2->WriteBack(CACHE_BASE::ACCESS_TYPE_LOAD);
}

/* ===================================================================== */

VOID StoreSingleFast(ADDRINT addr)
{
    const AccessResult dl1Result = dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_STORE);    
    if(!dl1Result.hit) dl2->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_STORE);    
    if(dl1Result.dirty_evicted) dl2->WriteBack(CACHE_BASE::ACCESS_TYPE_STORE);
}



/* ===================================================================== */

VOID Instruction(INS ins, void * v)
{
    if ( ! INS_IsStandardMemop(ins)) return;
    
    if (INS_MemoryOperandCount(ins) == 0) return;
    
    UINT32 readSize=0, writeSize=0;
    UINT32 readOperandCount=0, writeOperandCount=0;
    
    for (UINT32 opIdx = 0; opIdx < INS_MemoryOperandCount(ins); opIdx++)
    {
        if (INS_MemoryOperandIsRead(ins, opIdx))
        {
            readSize = INS_MemoryOperandSize(ins, opIdx);
            readOperandCount++;
            break;
        }
        if (INS_MemoryOperandIsWritten(ins, opIdx))
        {
            writeSize = INS_MemoryOperandSize(ins, opIdx);
            writeOperandCount++;
            break;
        }
    }
    
    if (readOperandCount > 0)
    {
        // map sparse INS addresses to dense IDs
        const ADDRINT iaddr = INS_Address(ins);
        const UINT32 instId = profile_l1.Map(iaddr);
        const UINT32 instId2 = profile_l2.Map(iaddr);

        const BOOL   single = (readSize <= 4);
                
        if( KnobTrackLoads )
        {
            if( single )
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE, (AFUNPTR) LoadSingle,
                    IARG_MEMORYREAD_EA,
                    IARG_UINT32, instId,
                    IARG_UINT32, instId2,
                    IARG_END);
            }
            else
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) LoadMulti,
                    IARG_MEMORYREAD_EA,
                    IARG_MEMORYREAD_SIZE,
                    IARG_UINT32, instId,
                    IARG_UINT32, instId2,
                    IARG_END);
            }
                
        }
        else
        {
            if( single )
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) LoadSingleFast,
                    IARG_MEMORYREAD_EA,
                    IARG_END);
                        
            }
            else
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) LoadMultiFast,
                    IARG_MEMORYREAD_EA,
                    IARG_MEMORYREAD_SIZE,
                    IARG_END);
            }
        }
    }
        
    if (writeOperandCount > 0)
    {
        // map sparse INS addresses to dense IDs
        const ADDRINT iaddr = INS_Address(ins);
        const UINT32 instId = profile_l1.Map(iaddr);            
        const UINT32 instId2 = profile_l2.Map(iaddr);            
        const BOOL   single = (writeSize <= 4);
                
        if( KnobTrackStores )
        {
            if( single )
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) StoreSingle,
                    IARG_MEMORYWRITE_EA,
                    IARG_UINT32, instId,
                    IARG_UINT32, instId2,
                    IARG_END);
            }
            else
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) StoreMulti,
                    IARG_MEMORYWRITE_EA,
                    IARG_MEMORYWRITE_SIZE,
                    IARG_UINT32, instId,
                    IARG_UINT32, instId2,
                    IARG_END);
            }
                
        }
        else
        {
            if( single )
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) StoreSingleFast,
                    IARG_MEMORYWRITE_EA,
                    IARG_END);
                        
            }
            else
            {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE,  (AFUNPTR) StoreMultiFast,
                    IARG_MEMORYWRITE_EA,
                    IARG_MEMORYWRITE_SIZE,
                    IARG_END);
            }
        }
            
    }
}

/* ===================================================================== */

VOID Fini(int code, VOID * v)
{
    // print D-cache profile_l1
    // @todo what does this print
    
    outFile << "PIN:MEMLATENCIES 1.0. 0x0\n";
            
    outFile <<
        "#\n"
        "# DCACHE L1 stats\n"
        "#\n";
    
    outFile << dl1->StatsLong("# ", CACHE_BASE::CACHE_TYPE_DCACHE);

    if( KnobTrackLoads || KnobTrackStores ) {
        outFile <<
            "#\n"
            "# LOAD stats\n"
            "#\n";
        
        outFile << profile_l1.StringLong();
    }

    outFile <<
        "#\n"
        "# DCACHE L2 stats\n"
        "#\n";
    
    outFile << dl2->StatsLong("# ", CACHE_BASE::CACHE_TYPE_DCACHE);

    if( KnobTrackLoads || KnobTrackStores ) {
        outFile <<
            "#\n"
            "# LOAD stats\n"
            "#\n";
        
        outFile << profile_l2.StringLong();
    }
    outFile.close();
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char *argv[])
{
    PIN_InitSymbols();

    if( PIN_Init(argc,argv) )
    {
        return Usage();
    }

    outFile.open(KnobOutputFile.Value().c_str());

    dl1 = new DL1::CACHE("L1 Data Cache", 
                         KnobL1CacheSize.Value() * KILO,
                         KnobL1LineSize.Value(),
                         KnobL1Associativity.Value());

    dl2 = new DL2::CACHE("L2 Data Cache", 
                         KnobL2CacheSize.Value() * KILO,
                         KnobL2LineSize.Value(),
                         KnobL2Associativity.Value());
    
    profile_l1.SetKeyName("iaddr          ");
    profile_l1.SetCounterName("dl1cache:miss       dl1cache:hit");
    profile_l2.SetKeyName("iaddr2          ");
    profile_l2.SetCounterName("dl2cache:miss       dl2cache:hit");

    COUNTER_HIT_MISS threshold;

    threshold[COUNTER_HIT] = KnobThresholdHit.Value();
    threshold[COUNTER_MISS] = KnobThresholdMiss.Value();
    
    profile_l1.SetThreshold( threshold );
    profile_l2.SetThreshold( threshold );
    
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns

    PIN_StartProgram();
    
    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
