#!/bin/bash

# Input Parameters
ARCHIVE_NAME=$1
if [ $ARCHIVE_NAME != "real" ] && [ $ARCHIVE_NAME != "synthetic" ] && [ $ARCHIVE_NAME != "A-SDF" ]
then
    echo "Unknown archive name ${ARCHIVE_NAME}. Use ./download_archives.sh [real|synthetic|A-SDF]"
    exit 0
fi
    

# Create directory
BASE_DIR="downloaded_archives"
DIR="${BASE_DIR}/${ARCHIVE_NAME}_parts/"
mkdir -p ${DIR}
echo "Created ${DIR} for saving"

PARTS=()
if [ $ARCHIVE_NAME == "real" ]
then
    for x in {a..r}
    do
        PARTS+=("a${x}")
    done
elif [ $ARCHIVE_NAME == "synthetic" ]
then
    for x in {a..c}
    do
        for y in {a..z}
        do
            PARTS+=("${x}${y}")
        done
    done
    for x in {a..y}
    do
        PARTS+=(".d${x}")
    done
elif [ $ARCHIVE_NAME == "A-SDF" ]
then
    for x in {a..q}
    do
        PARTS+=("a${x}")
    done  
fi

EVERYTHING_OK=1
# Download file
for PART in "${PARTS[@]}"
do
    echo "${DIR}"
    FILE_NAME="${ARCHIVE_NAME}.part.${PART}"
    URL="http://carto.cs.uni-freiburg.de/datasets/${ARCHIVE_NAME}_parts/${FILE_NAME}"
    #Check if file exists on the server
    if curl --output /dev/null --silent --head --fail "$URL"
    then
        echo "URL exists on server: $URL"
        # Download file
        if test -f "${DIR}/${FILE_NAME}"
        then
            echo "Skipping as file already exists locallly"
        else
            if wget -P ${DIR} ${URL}
            then
                echo "Successfully downloaded $URL"
            else
                echo "Error downloading $URL"
                EVERYTHING_OK = 0
            fi
        fi
    else
        echo "URL does not exist: $URL"
        EVERYTHING_OK = 0
    fi
done

# Unzip file
if [ $EVERYTHING_OK -eq 0 ]
then
    echo "Error downloading ${ARCHIVE_NAME} data"
    exit 1
fi

cat $DIR/* > "${BASE_DIR}/${ARCHIVE_NAME}.tar.gz"
echo "Successfully downloaded ${ARCHIVE_NAME}"
