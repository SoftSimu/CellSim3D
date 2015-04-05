#!/usr/bin/env bash
echoed=''
shopt -s nullglob
while [ TRUE ]; do

    # find unprocessed trajectory file
    trajFile=$(ls -tr traj*.xyz | head -n 1 )
    ext="${trajFile##*.}"
    #echo $ext
    if [ "$ext" != 'xyz' ]; then
        if [ -z $echoed ]; then
            echo -e "\rNo new trajectories to render..."
            echoed="yes"
        fi
        #break
        echo -ne "\roOO   "
        sleep 0.4
        echo -ne "\rOoO   "
        sleep 0.4
        echo -ne "\rOOo   "
        sleep 0.4
        echo -ne "\rOoO   "
        sleep 0.4
        echo -ne "\roOO   "
        sleep 0.1
        #sleep 3
    else
        echoed=''
        echo -e "\rFound $trajFile"
        echo "Waiting some time before rendering..."
        sleep 5
        # Create directory
        trajName="${trajFile%.*}"
        mkdir -p "$trajName/images" # 'already exists' message shouldn't happen
        # Start rendering with blender
        rm "$trajName/images"/* 2> /dev/null
        echo "Blender is rendering $trajName..."
        echo "You may monitor the $trajName/images folder to see progress"
        blender -b CellDiv.blend -Y -P render.py "$trajFile" "$trajName" &> /dev/null &
        pid=$!

        trap "kill $pid 2> /dev/null" EXIT

        tmp="unlikely string"
        echo -ne "oOO   "
        while kill -0 $pid 2> /dev/null; do
            echo -ne "\rOoO   "
            sleep 0.2
            echo -ne "\rOOo   "
            sleep 0.2
            echo -ne "\rOoO   "
            sleep 0.2
            echo -ne "\roOO   "
            sleep 0.2

            newestFile=$(ls -At "$trajName/images/" | head -n 1)

            if [ "$tmp" != "$newestFile" -a -n "$newestFile" ]; then
                echo -e "\rCreated $trajName/images/$newestFile"
                echo -ne "oOO   "
                tmp="$newestFile"
            fi
        done

        # Make the video
        imgPath="$trajName/images/CellDiv_%d.png"
        avconv -y -i "$imgPath" -r 60 -b 20k -vf scale=1920:1080 "$trajName/$trajName.mp4"
        pid=$!
        trap "kill $pid 2> /dev/null" EXIT
        echo -ne "oOO   "
        while kill -0 $pid 2> /dev/null; do
            echo -ne "\rOoO   "
            sleep 0.2
            echo -ne "\rOOo   "
            sleep 0.2
            echo -ne "\rOoO   "
            sleep 0.2
            echo -ne "\roOO   "
            sleep 0.2
        done
        mv "$trajFile" "$trajFile~"
        mv "$trajName.log" "$trajName/$trajName.log"
    fi
done
shopt -u nullglob
