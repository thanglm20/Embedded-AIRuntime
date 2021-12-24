    
# Introduction
    Branch r2.0
# Requirements 
    Refer: requirements.txt

# Build




Note: If you run on Snapdragon of QC
push this text to system/etc/mkshrc
export SNPE_ROOT=/data/snpe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SNPE_ROOT/lib
export PATH=$PATH:/system/bin
export ADSP_LIBRARY_PATH="$SNPE_ROOT/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
echo "[INFO] - ThangLMb: Export env for DSP\n"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vendor/lib


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib
export ADSP_LIBRARY_PATH="/data/local/tmp/snpe/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"


export PATH=$PATH:/data/local/tmp/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/lib

export SNPE_ROOT=/data/local/tmp/snpe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SNPE_ROOT/lib
export ADSP_LIBRARY_PATH="$SNPE_ROOT/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
