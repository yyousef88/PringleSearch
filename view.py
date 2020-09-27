from flask import Flask, render_template, redirect, url_for, request
from forms import QueryForm
from controller import Controller
import os
import time

SECRET_KEY = os.urandom(32)
app = Flask(__name__) 
app.config['SECRET_KEY'] = SECRET_KEY
controller = Controller()

#main page from where you deploy search
@app.route('/', methods = ["GET", "POST"])
def search():
    form = QueryForm()
    if form.validate_on_submit():
        query = request.form['query']
        return redirect(url_for('results', query = query, page = 0))
    return render_template("index.html", form = form)

#results page
@app.route('/results/<query>/page-<page>')
def results(query,page):
    page = int(page)
    form = QueryForm()
    controller.setQuery(query)
    #times how long it takes to retrieve urls
    t1 = time.time()
    urls = controller.retrieveUrls()
    t2 = time.time()
    #splits pages into 10
    urls = split(urls, 10)[page:page+10]
    page_indices = range(len(urls))
    return render_template("search_results.html", urls = urls[page] if len(urls) > 0 else ["No results"], form=form, pages = page_indices, query = query, time = round(t2-t1, 3))

#splits list into equal n-sized lists
def split(l, n):
    x = []
    for i in range(0, len(l), n):
        x.append(l[i:i + n])
    return x

app.run()
