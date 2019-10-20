#!/bin/bash

start=1
end=1

#parameters
    sizemin=1
    sizemax=6   #3 6 16 32
    ply_in_name="stipples/ima_min${sizemin}_max${sizemax}.0001.ply"
    image_in_name="stipples/ima_crop.0001.png"
    
    #strokes length
    maxCount=60
    
    #use flags booleans
    dopixelforce=1
    donoiseforce=1
    dogradientforce=0
    dotangentforce=0
    doboundforce=0
    
    #forces strength
    pixelInfluence=10
    noiseInfluence=5
    gradientInfluence=0
    tangentInfluence=0
    boundForceFactor=0
    dragInfluence=0 #drag : dampens speed at each strokes
    
    #pixelforce
    minpixforce=0
    halfperception=2 #1->3x3 filter 2->5x5 3->7x7 etc ..
    
    #noise : possible noise types
    #0 Value 
    #1 ValueFractal 
    #2 Perlin 
    #3 PerlinFractal 
    #4 Simplex 
    #5 SimplexFractal 
    #6 Cellular 
    #7 WhiteNoise 
    #8 Cubic 
    #9 CubicFractal 
    zstep=0.01  #rate of change for third dimension of noise
    fnnoisetype=3
    fnfrequency=0.1
    docurlnoise=0   #compute curl noise
    
    #gradient and/or tangent force
    gradientblur=3  #must be an even number

    
    #bound force : push points inwards the frame
    bound=20    #width of active boundary in pixels
    
    #limiters
    colinearlimit=1     #avoid straight lines if true
    maxSpeed=10000      #clamps speed
    
    #rendering
    lineopacity=0.4
    colorpower=2    #fade to white along the stroke
    
    linewidthmin=1
    linewidthmax=10
    splinestep=0.01  #-1: lines 0: auto
    oversampling=3
    output_x=1920
    debug=0 #output pictures visualizing various parameters
    
    image_out_name="stipples/result_max${sizemax}_count${maxCount}_pix${pixelInfluence}_noise${noiseInfluence}.png"
    
for i in `seq $start $end`;
do
    start_time=$SECONDS
    ii=$(printf "%04d" $i)
    echo "iteration : $ii"
    ./build/stipplecurves $ply_in_name $image_in_name $image_out_name $maxCount $dopixelforce $pixelInfluence $minpixforce $halfperception $dogradientforce $gradientblur $gradientInfluence $dotangentforce $tangentInfluence $dragInfluence $donoiseforce $docurlnoise $noiseInfluence $zstep $fnnoisetype $fnfrequency $doboundforce $bound $boundForceFactor $colinearlimit $maxSpeed $lineopacity $colorpower $sizemin $sizemax $linewidthmin $linewidthmax $splinestep $oversampling $output_x $debug
    elapsed_time=$(($SECONDS - $start_time))
    echo "scribbled in : $(($elapsed_time/60)) min $(($elapsed_time%60)) sec"  
done    
