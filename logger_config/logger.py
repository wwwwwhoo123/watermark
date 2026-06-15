import os
import csv
import logging


def logger_config(log_path):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level=logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def update_csv_row(csv_file_path, imagename, attacker_results):
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    row_updated = False
    for row in rows:
        if row[0] == imagename:
            row[-len(attacker_results):] = attacker_results
            row_updated = True
            break

    if not row_updated:
        print(f'Warning: No row found for image {imagename}.')
        return

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
