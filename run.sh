#!/usr/bin/env bash


source env/bin/activate

export FLASK_APP=src
export FLASK_ENV=development

flask run --eager-loading

# if '__main__' == __name__:
#    app.run(host= "0.0.0.0")
# python3 -m medit.__init__
