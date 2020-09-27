from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

#creates two fields for use with flask
class QueryForm(FlaskForm):
    query = StringField("", [DataRequired()])
    submit = SubmitField("Search")
    
