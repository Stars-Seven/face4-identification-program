"画像収集プログラム"
from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": './haru'})
crawler.crawl(keyword='波留', max_num=200)  #max_num=上限1000 まで可能