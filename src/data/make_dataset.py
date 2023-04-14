import os
import requests
import time
import random
import shutil
import math
import json
import jsonlines
from PIL import Image
from re import sub

import pandas as pd

DATA_FP = '../../data'
MEDIUMS = ['oil', 'watercolor', 'pastel', 'pencil', 'tempera']
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/50.0.2661.102 Safari/537.36'}
VERSION = 2


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
    time.sleep(1)

    if image_data.status_code == 200:
        create_dir(save_fp)
        with open(f"{save_fp}/{image_name}.jpg",
                  'wb') as file_obj:
            file_obj.write(image_data.content)


# TODO: Check for multiple mediums
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


def clean_string_v2(dirty_str):
    return sub('\W+', '', dirty_str.replace(' ', '_').lower())


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
                url = json.loads(
                    requests.get(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}').text)[
                    'primaryImageSmall']
                if url == '':
                    print(f'URL is empty for Object ID {obj_id}, continuing...')
                    continue
            except:
                print(f'Error occured obtaining url for Objext ID {obj_id}, continuing...')
                continue

            save_fp = f"{DATA_FP}/processed/metropolitan/{medium}"

            artist = clean_string_v2(str(row['Artist Display Name']))

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
            artist = clean_string_v2(row['Artist'])

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
            url = row['iiifurl']
            save_fp = f"{DATA_FP}/processed/national_goa/{info_dict[obj_id]['medium']}"
            artist = clean_string_v2(info_dict[obj_id]['attribution'])
            year = str(info_dict[obj_id]['endyear']).split('.')[0]

            image_name = f'{artist}_{year}'
            handle_image_data(url + '/full/!20000,20000/0/default.jpg', save_fp, image_name)

        if i % 100 == 0:
            print(f'{i} iterations completed out of {len(info_csv)}')


def handle_chicago():
    """
    Structure the Art Institute of Chicago dataset.
    """
    i = 0

    with jsonlines.open('../../data/external/allArtworks.jsonl') as file_obj:
        num_lines = [line for line in file_obj.iter()]

    with jsonlines.open('../../data/external/allArtworks.jsonl') as file_obj:
        for line in file_obj.iter():
            # Link used to get necessary info to the image
            info_url = f"https://api.artic.edu/api/v1/artworks/{line['id']}?fields=id,title,image_id,term_titles,fiscal_year"
            info_resp = requests.get(info_url, headers=HEADERS).content.decode()

            info_resp = json.loads(info_resp)

            # Medium descriptions lie in here
            terms = info_resp['data']['term_titles']
            for term in terms:
                medium = handle_medium(term)
                if medium is not False:
                    break

            if medium is False:
                continue

            artist = clean_string_v2(line['artist_title'])
            year = info_resp['data']['fiscal_year']
            image_name = f'{artist}_{year}'

            image_id = info_resp['data']['image_id']
            config_url = info_resp['config']['iiif_url']

            save_fp = f"{DATA_FP}/processed/chicago/{medium}"
            image_url = f'{config_url}/{image_id}/full/843,/0/default.jpg'
            handle_image_data(image_url, save_fp, image_name)

            if i % 100 == 0:
                print(f'{i} iterations completed out of {len(num_lines)}')

            i += 1


def copy_images(image_list, dest):
    """
    Copies an image from source directory to a destination directory.
    """
    create_dir(dest)
    for image_path in image_list:
        image_name = image_path.split('/')[-1]
        shutil.copy(image_path, f'{dest}/{image_name}')


def get_test_images():
    """
    Obtains all the testing images previously used, ensuring they do not make it back into our training dataset.
    """
    res_dict = {'oil': [], 'watercolor': [], 'tempera': [], 'pastel': [], 'pencil': []}
    # TODO: Only checks v1 as of now
    des_fp = '../../data/processed/v1/test'

    for medium in os.listdir(des_fp):
        for image in os.listdir(f'{des_fp}/{medium}'):
            res_dict[medium].append(image)

    return res_dict


def patch_images():
    """
    Will split original images into four sub-images (non-overlapping). It will do this with the version provided, and
    create a new version with the patching implemented.
    """
    root_dir = f'../../data/processed/v{VERSION}/'
    for data_dir in os.listdir(root_dir):
        data_fp = os.path.join(root_dir, data_dir)
        for medium_dir in os.listdir(data_fp):
            medium_fp = os.path.join(data_fp, medium_dir)

            new_version_fp = f'../../data/processed/v{VERSION + 1}/{data_dir}/{medium_dir}'
            if not os.path.exists(f'{new_version_fp}'):
                create_dir(f'{new_version_fp}')

            for image_name in os.listdir(medium_fp):
                image = Image.open(os.path.join(medium_fp, image_name))

                width, height = image.size
                crop_size = (width // 2, height // 2)

                for i in range(2):
                    for j in range(2):
                        left = j * crop_size[0]
                        top = i * crop_size[1]
                        right = (j + 1) * crop_size[0]
                        bottom = (i + 1) * crop_size[1]

                        cropped_image = image.crop((left, top, right, bottom))

                        cropped_image.save(f"{new_version_fp}/{image_name.split('.jpg')[0]}_{i}_{j}.jpg")


def make_data_split():
    """
    Creates the training, validation, and testing data sets.
    """
    # Contains images used in all prior testing datasets
    test_images = get_test_images()
    image_dict = {'oil': [], 'watercolor': [], 'pencil': [], 'tempera': [], 'pastel': []}
    data_dirs = ['modern_art', 'national_goa', 'metropolitan']

    for directory in data_dirs:
        mediums_lst = [medium for medium in os.listdir(f'{DATA_FP}/processed/{directory}')]
        for medium in mediums_lst:
            medium_path = f'{DATA_FP}/processed/{directory}/{medium}'
            images = [f'{medium_path}/{image}' for image in os.listdir(medium_path) if image not in test_images[medium]]
            prior_test_images = [f'{medium_path}/{image}' for image in os.listdir(medium_path) if
                                 image in test_images[medium]]

            random.shuffle(images)
            test_images[medium].extend(prior_test_images)
            image_dict[medium].extend(images)

    least_images = math.inf
    for medium, image_lst in image_dict.items():
        if len(image_lst) < least_images:
            least_images = len(image_lst)

    for medium, image_lst in image_dict.items():
        # Even distribution, give 10 percent to validation and testing datasets
        image_lst = image_lst[0:least_images]

        num_images = len(image_lst)
        train_slice = int(math.floor(num_images * 0.8))
        train_lst = image_lst[0:train_slice]

        val_slice = int(train_slice + (math.floor(num_images * 0.1)))
        val_lst = image_lst[train_slice:val_slice]

        test_lst = image_lst[val_slice:]

        copy_images(train_lst, f'{DATA_FP}/processed/v{VERSION}/train/{medium}')
        copy_images(test_lst, f'{DATA_FP}/processed/v{VERSION}/val/{medium}')
        copy_images(val_lst, f'{DATA_FP}/processed/v{VERSION}/test/{medium}')


def main():
    """
    Controls this script.
    """
    # make_data_split()
    patch_images()
    # handle_national_goa()


if __name__ == '__main__':
    main()
