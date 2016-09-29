FROM floydhub/dl-docker:cpu

RUN luarocks install dp && luarocks install mnist 
RUN wget https://raw.githubusercontent.com/rtsisyk/luafun/master/fun-scm-1.rockspec && luarocks install fun-scm-1.rockspec

WORKDIR /code
CMD ["th"]
