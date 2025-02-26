#!/bin/bash

# Если пераметры не переданы - выводим справку:
if [ $# -lt 1 ]; then
    echo 'Разрезает медиафайл на части, не превышающие заданного размера.'
    echo 'Использование:'
    echo './split-video.sh FILE SIZELIMIT "FFMPEG_ARGS"'
    echo 
    echo 'Параметры'
    echo '    - FILE:        Путь до исходного файла, подлежащего разрезанию (обязательный параметр)'
    echo '    - SIZELIMIT:   Максимальный размер каждой части в байтах (по-умолчанию 1000000000)'
    echo '    - FFMPEG_ARGS: Параметры, передаваемые непосредственно в ffmpeg (по-умолчанию "-c copy")'
    echo 
    echo 'В основе скрипта лежит код, предложенный в https://superuser.com/a/714749'
    exit 0
fi

FILE="$1"

# Читаем параметры ffmpeg из CLI или берём значение по-умолчанию:
if [ $# -lt 3 ]; then
    FFMPEG_ARGS="-c copy"
else
    FFMPEG_ARGS="$3"
fi

# Читаем размер фрагментов из CLI или берём значение по-умолчанию:
if [ $# -lt 2 ]; then
    SIZELIMIT=1000000000
else
    SIZELIMIT="$2"
fi

# Duration of the source video
DURATION=$(ffprobe -i "$FILE" -show_entries format=duration -v quiet -of default=noprint_wrappers=1:nokey=1|cut -d. -f1)

# Duration that has been encoded so far
CUR_DURATION=0

# Filename of the source video (without extension)
BASENAME="${FILE%.*}"

# Extension for the video parts
#EXTENSION="${FILE##*.}"
EXTENSION="mp4"

# Number of the current video part
i=1

# Filename of the next video part
NEXTFILENAME="$BASENAME-$i.$EXTENSION"

echo "Duration of source video: $DURATION"

# Until the duration of all partial videos has reached the duration of the source video
while [[ $CUR_DURATION -lt $DURATION ]]; do
    # Encode next part
    echo ffmpeg -i "$FILE" -ss "$CUR_DURATION" -fs "$SIZELIMIT" $FFMPEG_ARGS "$NEXTFILENAME"
    ffmpeg -ss "$CUR_DURATION" -i "$FILE" -fs "$SIZELIMIT" $FFMPEG_ARGS "$NEXTFILENAME"

    # Duration of the new part
    NEW_DURATION=$(ffprobe -i "$NEXTFILENAME" -show_entries format=duration -v quiet -of default=noprint_wrappers=1:nokey=1|cut -d. -f1)

    # Total duration encoded so far
    CUR_DURATION=$((CUR_DURATION + NEW_DURATION))

    i=$((i + 1))

    echo "Duration of $NEXTFILENAME: $NEW_DURATION"
    echo "Part No. $i starts at $CUR_DURATION"

    NEXTFILENAME="$BASENAME-$i.$EXTENSION"
done