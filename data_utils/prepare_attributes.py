import argparse
import os
from loguru import logger
import pandas as pd

def list2txt(name, attribute_list, output_folder):
    file_name = os.path.join(output_folder, f"{name}.txt")
    with open(file_name, 'w') as file:
        for attribute in attribute_list:
            file.write(f"{attribute}\n")

def process_proper_noun(attribute):
    words = attribute.split(" ")
    processed_words = []
    for w in words:
        if w != 'the':
            processed_words.append(w[0].upper() + w[1:].lower())
        else:
            processed_words.append(w)
    return " ".join(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default="attributes_source")
    parser.add_argument('-o', '--ouput_folder', default="attributes")
    args = parser.parse_args()
    logger.info("Start process the attributes")

    logger.info("Processing birthplace...")
    bp_file = os.path.join(args.input_folder, "worldcities.xlsx")
    bp_df = pd.read_excel(bp_file)
    cities = list(set(bp_df['city']))
    cities = [process_proper_noun(c) for c in cities]
    list2txt("city", cities, args.ouput_folder)
    logger.info(f"{len(cities)} birthplace saved to {args.ouput_folder}")

    logger.info("Processing hobbies...")
    hb_file = os.path.join(args.input_folder, 'hobbylist.csv')
    hb_df = pd.read_csv(hb_file)
    hobbies = list(set(hb_df['Hobby-name']))
    hobbies = [h.lower() for h in hobbies]
    list2txt("hobby", hobbies, args.ouput_folder)
    logger.info(f"{len(hobbies)} hobbies saved to {args.ouput_folder}")

    logger.info("Processing Universities...")
    uni_file = os.path.join(args.input_folder, 'World University Rankings 2023.csv')
    uni_df = pd.read_csv(uni_file)
    allowed_rank = [str(i) for i in range(200)] + ['401–500', '501–600', '251–300', '351–400', '201–250', '301–350']
    uni_df = uni_df[uni_df['University Rank'].isin(allowed_rank)]
    unis = list(set(uni_df['Name of University']))
    unis = [process_proper_noun(u) for u in unis]
    list2txt("University", unis, args.ouput_folder)
    logger.info(f"{len(unis)} universities saved to {args.ouput_folder}")

    logger.info("Processing major...")
    maj_file = os.path.join(args.input_folder, 'recent-grads.csv')
    maj_df = pd.read_csv(maj_file)
    majors = list(set(maj_df['Major']))
    majors = [m.lower() for m in majors]
    list2txt("major", majors, args.ouput_folder)
    logger.info(f"{len(majors)} majors saved to {args.ouput_folder}")

    logger.info("Processing top 500 popular cities as workcity...")
    maj_df_sorted = bp_df.sort_values(by='population', ascending=False)
    work_cities = list(maj_df_sorted['city'].unique()[:500])
    list2txt("Workcity", work_cities, args.ouput_folder)
    logger.info(f"{len(work_cities)} work cities saved to {args.ouput_folder}")

    logger.info(f"All process finished")
