import os
import requests
import time
import random
import shutil
import math
import json

import pandas as pd

DATA_FP = '../../data'
MEDIUMS = ['oil', 'watercolor', 'pastel', 'pencil', 'tempera']
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/50.0.2661.102 Safari/537.36'}


# TODO: Save photos by something more descriptive

# TODO: Things we'd like to analyze
#   Number of mediums (If we don't have as high of an accuracy, we do have stats for other mediums?)
#   Number of artists
#   When the artwork was created? Newer vs. older artwork may have an effect?
#   How many samples in your dataset that you have


def create_dir(des_fp: str):
    """
    Iteratively creates a file path.
    """
    if des_fp is None:
        raise ValueError(f'Expecting a valid file path, got: {des_fp}')

    # Iteratively check if the file path exists
    split_path = des_fp.split('/')
    curr_path = ''
    for i in range(len(split_path)):
        curr_path += split_path[i] + '/'
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)


def handle_image_data(url, save_fp, image_name):
    """
    Retrieves an image from the World Wide Web given a URL.
    """
    image_data = requests.get(url, headers=HEADERS)
    time.sleep(0.1)

    if image_data.status_code == 200:
        create_dir(save_fp)
        with open(f"{save_fp}/{image_name}.jpg",
                  'wb') as file_obj:
            file_obj.write(image_data.content)


def handle_medium(raw_medium: str):
    """
    Given a sentence describing the medium used, identify the medium and make it one word.
    """
    if str(raw_medium) == 'nan' or raw_medium is None:
        return False

    raw_medium = raw_medium.lower()

    for curr_med in MEDIUMS:
        if curr_med in raw_medium:
            return curr_med

    return False


def clean_string(dirty_str: str):
    """
    Formats a string to snakecase.
    """
    clean_str = dirty_str.lower().replace(',', '').replace(' ', '_').replace('.', '').replace('-', '_').replace("'", '')
    return clean_str


def handle_wiki_art():
    """
    Structure the WikiArt dataset.
    """
    pass


def handle_image_net():
    """
    Structure the ImageNet dataset.
    """
    pass

def handle_metropolitan_moa():
    """
    Structure the metropolitan museum of art dataset.
    """
    csv = pd.read_csv(f'{DATA_FP}/external/MetObjects.csv')
    for i, row in csv.iterrows():
        medium = handle_medium(row['Medium'])

        if medium is not False and str(row['Is Public Domain']) == "True":
            obj_id = row['Object ID']
            try:
                url = json.loads(requests.get(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{ obj_id }').text)['primaryImageSmall']
                if url == '':
                    print(f'URL is empty for Object ID {obj_id}, continuing...')
                    continue
            except:
                print('Error occured obtaining url for Objext ID {obj_id}, continuing...')
                continue

            save_fp = f"{DATA_FP}/processed/metropolitan/{medium}"

            artist = str(row['Artist Display Name'])

            year = row['Object End Date']

            if '/' in artist:
                artist = artist.split('/')[1]
            if '|' in artist:
                artist = artist.split('|')[1]

            image_name = f'{artist}_{year}'
            handle_image_data(url, save_fp, image_name)

        if i % 100 == 0:
            print(f'{i} iterations completed out of {len(csv)}')

def handle_modern_art():
    """
    Structure the museum of modern art dataset.
    """
    csv = pd.read_csv(f'{DATA_FP}/external/Artworks.csv')

    for i, row in csv.iterrows():
        medium = handle_medium(row['Medium'])

        if medium is not False:
            url = row['ThumbnailURL']
            if str(url) == 'nan':
                continue

            save_fp = f"{DATA_FP}/processed/modern_art/{medium}"
            artist = clean_string(row['Artist'])

            year = row['EndDate']
            if year == '(0)':
                year = row['BeginDate']
            # There were multiple years in some of these?
            if len(year) > 6:
                year = year.split(' ')[0]
            year = year.strip('(').strip(')')

            if '/' in artist:
                artist = artist.split('/')[1]
            image_name = f'{artist}_{year}'
            handle_image_data(url, save_fp, image_name)

        if i % 100 == 0:
            print(f'{i} iterations completed out of {len(csv)}')


def handle_national_goa():
    """
    Structure the National Gallery of Art's dataset.
    """
    images_csv = pd.read_csv(f'{DATA_FP}/external/published_images.csv')
    info_csv = pd.read_csv(f'{DATA_FP}/external/objects.csv')

    info_dict = dict()
    for i, row in info_csv.iterrows():
        medium = handle_medium(row['medium'])

        if medium is not False:
            obj_id = row['objectid']
            info_dict[obj_id] = row.to_dict()
            info_dict[obj_id]['medium'] = medium
            info_dict[obj_id].pop('objectid')

        if i % 100 == 0:
            print(f'{i} iterations completed out of {len(info_csv)}')

    print('\n')

    for i, row in images_csv.iterrows():
        obj_id = row['depictstmsobjectid']
        if obj_id in info_dict.keys():
            url = row['iiifthumburl']
            save_fp = f"{DATA_FP}/processed/national_goa/{info_dict[obj_id]['medium']}"
            artist = clean_string(info_dict[obj_id]['attribution'])
            year = str(info_dict[obj_id]['endyear']).split('.')[0]

            image_name = f'{artist}_{year}'
            handle_image_data(url, save_fp, image_name)

        if i % 100 == 0:
            print(f'{i} iterations completed out of {len(info_csv)}')


def copy_images(src, dest, image_list):
    """
    Copies an image from source directory to a destination directory.
    """
    create_dir(dest)
    for image_name in image_list:
        shutil.copy(f'{src}/{image_name}', f'{dest}/{image_name}')


def make_data_split():
    """
    Creates the training, validation, and testing data sets.
    """
    des_dirs = ['modern_art', 'national_goa']
    check_dict = dict()
    for curr_dir in des_dirs:
        mediums = [medium for medium in os.listdir(f'{DATA_FP}/processed/{curr_dir}')]
        check_dict[curr_dir] = dict()
        for curr_medium in mediums:
            medium_path = f'{DATA_FP}/processed/{curr_dir}/{curr_medium}'
            images = [image for image in os.listdir(medium_path)]
            random.shuffle(images)

            # (About) eighty percent of the images shall go to training
            curr_len = len(images)

            first_slice = int(math.floor(curr_len * 0.8))
            train_lst = images[0:first_slice]

            # (About) ten percent of the images shall go to validation
            second_slice = int(first_slice + (math.floor(curr_len * 0.1)))
            val_lst = images[first_slice:second_slice]

            # The rest of the images will go to testing (should be around ten percent)
            test_lst = images[second_slice:]

            copy_images(medium_path, f'{DATA_FP}/processed/v1/train/{curr_medium}', train_lst)
            copy_images(medium_path, f'{DATA_FP}/processed/v1/val/{curr_medium}', val_lst)
            copy_images(medium_path, f'{DATA_FP}/processed/v1/test/{curr_medium}', test_lst)


def main():
    """
    Controls this script.
    """
    # handle_wiki_art()
    # make_data_split()
    handle_metropolitan_moa()


if __name__ == '__main__':
    main()
