# -*- coding: utf-8 -*-

# adapted from http://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search
# this is much better: https://github.com/NikolaiT/incolumitas/blame/master/content/Programming/googlesearch-a-rapid-python-class-to-get-search-results.md

import re
import os
import argparse
import sys
import json
import logging
import boto
import boto.s3
from boto.s3.key import Key
import StringIO
from bs4 import BeautifulSoup
import pandas

#will store them in amazon S3
AWS_ACCESS_KEY_ID = "AKIAJYGB4P7S4ZF2***"
AWS_SECRET_ACCESS_KEY = "w8beNXSoXGuaHYXK***" #sorry, private
AWS_S3_BUCKET = "dlhk-flower-app"

boto.config.add_section('Boto')
boto.config.set('Boto','http_socket_timeout','3')

try:
  # For Python 3.0 and later
  from urllib.request import urlopen, Request
except ImportError:
  # Fall back to Python 2's urllib2
  from urllib2 import urlopen, Request

FORMAT = "[%(levelname)s] %(filename)s::%(funcName)s():%(lineno)s | %(message)s"
logger = logging.getLogger()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=FORMAT)

IMAGESIZE = 'isz:lt,islt:qsvga' #larger than 400x300; good to remove icons
PHOTO = 'itp:photo'
COLOR = 'ic:color'
JPGONLY = 'ift:jpg'
#usage rights in excess of fair use / fair dealing
#https://support.google.com/websearch/answer/29508?hl=en
#n: standard fair use
#f: free to use or share non-commercially
#fc: free to use or share even commercially
#fm: free to use, share or modify non-commercially
#fmc:free to use, share or modify even commercially
LEGALTERMS = ['n', 'f', 'fc', 'fm', 'fmc']

def upload(file_object, filename):
  conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, debug=0)
  bucket = conn.get_bucket(AWS_S3_BUCKET)
  k = Key(bucket)
  k.key = filename    # In my situation, ids at the end are unique
  fp = StringIO.StringIO(file_object.read())   # Wrap object
  k.set_contents_from_file(fp)
  k.set_acl('public-read')

def setLevel(level):
  logger = logging.getLogger()
  logger.setLevel(level)

def get_soup(url, headers):
  return BeautifulSoup(urlopen(Request(url, headers=headers)), 'html.parser')

def get_image(path, query, max_images, legal_term='n', upload_to_s3=False):
  qry = query.split()
  qry = "+".join(qry)
  url = "https://www.google.com.hk/search?q=" + qry + "+花&source=lnms&tbm=isch&num=100"
  url += '&tbs=' + JPGONLY + ',' + PHOTO + ',' + COLOR #augment with filters that give us only color photos in jpeg format
  if legal_term != 'n':
    #url += '&tbs=sur:' + legal_term
    url += ',sur:' + legal_term

  logging.info('attempting query for search term: ['+ query + '] url = ' + url)
  headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0"}
  soup = get_soup(url, header)
  ActualImages = []
  for a in soup.find_all("div", {"class":"rg_meta"}):
    #link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
    link = json.loads(a.text)["ou"]
    #ActualImages.append((link, Type))
    ActualImages.append(link)
  logging.info('found ' + str(len(ActualImages)) + ' images for this query...getting first ' + str(max_images) + ' images')
  i = 1
  #for (img, Type) in ActualImages:
  for img in ActualImages:
    try:
      req = Request(url=img, headers=headers)
      raw_img = urlopen(req)
      filename = "img" + "_" + str(i) + ".jpg"

      if upload_to_s3:
        logging.info("  attempting upload to S3: dataset/" + filename)
        upload(raw_img, "dataset/" + query + "/" + filename)
      else:
        newpath = os.path.join(path, query)
        if not os.path.exists(newpath):
          os.makedirs(newpath)
        img_path = os.path.join(newpath, filename)
        logging.info('  attempting to save file:' + img_path)
        f = open(img_path, 'wb')
        f.write(raw_img.read())
        f.close()

      i += 1
      if i > max_images:
        break
    except Exception as e:
      logging.error("could not retrieve: " + img)
      logging.error(e)

if __name__ == '__main__':
  setLevel(level=logging.DEBUG)
  parser = argparse.ArgumentParser(description='Scrape Google images')
  parser.add_argument('-n', '--num_images', default=5, type=int, help='num images to save per query')
  parser.add_argument('-p', '--path', default='/mnt/scratch/flowers/images/', type=str, help='data root path')
  parser.add_argument('-s', '--s3', help='upload to s3', action='store_true')
  parser.add_argument('-l', '--legal',
                      default='n', type=str, choices=LEGALTERMS,
                      help='download based on legal use rights: {'+ ','.join(LEGALTERMS)  +'}')
  parser.add_argument('-f', '--file',
                      default='./query.csv', type=str,
                      help='the universe of flowers in csv format.  uses column `search_term`')
  args = parser.parse_args()

  univ = pandas.read_csv(args.file)
  try:
    for q in univ.search_term:
      get_image(args.path, q, args.num_images, args.legal, args.s3)
  except KeyboardInterrupt:
    pass
  sys.exit()
