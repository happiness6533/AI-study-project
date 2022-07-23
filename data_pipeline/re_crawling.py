# 2.4.4 & 2.4.5 Re and Beautiful Soup
# Re 함수
import re

pattern = ' \W+'
re_pattern = re.compile(pattern)
pattern = ' \W+'
re_pattern = re.compile(pattern)
re.search("(\w+)", "wow, it is awesome")
re.split("(\w+)", "wow, it is world of word")
re.sub("\d", "number", "7 candy")
# Beautiful soup 사용
# pip install html5lib 으로 html5lib 를 추가 설치

from bs4 import BeautifulSoup

string = '<body> 이 글은 Beautiful soup 라이브러리를 사용하는 방법에 대한 글입니다. <br> </br> 라이브러리를 사용하면 쉽게 HTML 태그를 제거할 수 있습니다.</body>'
string = BeautifulSoup(string, "html5lib").get_text()  # HTML 태그를 제외한 텍스트만 가져온다
print(string)
