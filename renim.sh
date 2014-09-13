#!/usr/bin/env bash

# First get the trajectories to render
rm .render_queue &> /dev/null
shopt -s nullglob
for file in traj*.xyz; do
    if [ -z $(grep -o "$file" .render_done &> /dev/null) ]; then
        echo "$file" >> .render_queue
    fi
done
shopt -u nullglob # revert nullglob back to it's normal default status

while [ TRUE ]; do

    # Render the queued trajectories
    while read trajFile || [[ -n $line ]]; do
        # Create directory
        trajName="${trajFile%.*}"
        mkdir -p "$trajName/images" # 'already exists' message shouldn't happen
        # Start rendering with blender
        echo "Now processing $trajName" > .now_proc
        rm "$trajName/images"/*
        echo "Blender is rendering $trajName..."
        echo "you may monitor the $trajName/images folder to see progress"
        blender -b CellDiv.blend -Y -P render.py "$trajFile" "$trajName" &> /dev/null &
        pid=$!

        trap "kill $pid 2> /dev/null" EXIT

        tmp="unlikely string"
        echo -ne "|"
        while kill -0 $pid 2> /dev/null; do
            echo -ne "\b\r/"
            sleep 0.2
            echo -ne "\b\r--"
            sleep 0.2
            echo -ne "\b\r\\ "
            sleep 0.2
            echo -ne "\b\r|"
            sleep 0.2

            newestFile=$(ls -t "$trajName/images/" | head -n 1)
            
            if [ "$tmp" != "$newestFile" -a -n "$newestFile" ]; then
                echo -e "\b\rCreated $trajName/images/$newestFile"
                echo -ne "|"
                tmp="$newestFile"
            fi
        done

        # Make the video
        imgPath="$trajName/images/CellDiv_%d.png"
        avconv -i "$imgPath" "$trajName/$trajName.mp4" &> /dev/null &
        pid=$!
        trap "kill $pid 2> /dev/null" EXIT
        echo -ne "|"
        while kill -0 $pid 2> /dev/null; do
            echo -ne "\b\r/"
            sleep 0.2
            echo -ne "\b\r--"
            sleep 0.2
            echo -ne "\b\r\\ "
            sleep 0.2
            echo -ne "\b\r|"
            sleep 0.2
        done

        # Add name to the done file

        echo "$trajFile" >> .render_done
    done < .render_queue

    # Now check if any new trajectories were created
    rm .render_queue
    
    for file in traj*.xyz; do
        if [ -z $(grep -o "$file" .render_done) ]; then  # If this file hasn't been processed yet
                echo "$file" >> .render_queue
        fi
    done

    if [ ! -e .render_queue ]; then
        break
    fi
        
done
