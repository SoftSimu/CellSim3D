#!/usr/bin/env bash
echoed=''

while [ TRUE ]; do

    # find unprocessed trajectory file
    trajFile=$(ls -tr traj*.xyz 2> /dev/null | head -n 2> /dev/null)
    if [ -z $trajFile ]; then
        if [ -z $echoed ]; then
            echo "No new trajectories to render..."
            echoed="yes"
        fi
        #break
        echo -ne "\b\b\b\b\boOO  "
        sleep 0.4
        echo -ne "\b\b\b\b\bOoO  "
        sleep 0.4
        echo -ne "\b\b\b\b\bOOo  "
        sleep 0.4
        echo -ne "\b\b\b\b\bOoO  "
        sleep 0.4
        echo -ne "\b\b\b\b\boOO  "
        sleep 0.4
        #sleep 3 
    else
        echoed=''
        echo "Found $trajFile"        
        # Create directory
        trajName="${trajFile%.*}"
        mkdir -p "$trajName/images" # 'already exists' message shouldn't happen
        # Start rendering with blender
        rm "$trajName/images"/* 2> /dev/null
        echo "Blender is rendering $trajName..."
        echo "you may monitor the $trajName/images folder to see progress"
        blender -b CellDiv.blend -Y -P render.py "$trajFile" "$trajName" &> /dev/null &
        pid=$!

        trap "kill $pid 2> /dev/null" EXIT

        tmp="unlikely string"
        echo -ne "oOO   "
        while kill -0 $pid 2> /dev/null; do
            echo -ne "\b\b\b\b\b\bOoO   "
            sleep 0.2
            echo -ne "\b\b\b\b\b\bOOo   "
            sleep 0.2
            echo -ne "\b\b\b\b\b\bOoO   "
            sleep 0.2
            echo -ne "\b\b\b\b\b\boOO   "
            sleep 0.2

            newestFile=$(ls -t "$trajName/images/" | head -n 1)
            
            if [ "$tmp" != "$newestFile" -a -n "$newestFile" ]; then
                echo -e "\b\rCreated $trajName/images/$newestFile"
                echo -ne "oOO"
                tmp="$newestFile"
            fi
        done

        # Make the video
        imgPath="$trajName/images/CellDiv_%d.png"
        avconv -i "$imgPath" "$trajName/$trajName.mp4" &> /dev/null &
        pid=$!
        trap "kill $pid 2> /dev/null" EXIT
        echo -ne "oOO   "
        while kill -0 $pid 2> /dev/null; do
            echo -ne "\b\b\b\b\b\bOoO   "
            sleep 0.2
            echo -ne "\b\b\b\b\b\bOOo   "
            sleep 0.2
            echo -ne "\b\b\b\b\b\bOoO   "
            sleep 0.2
            echo -ne "\b\b\b\b\b\boOO   "
            sleep 0.2
        done
        mv "$trajFile" "$trajFile~"
    fi
done
