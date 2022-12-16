const Handlebars = require("handlebars");
const Bootstrap = require("bootstrap")

import template from './templates/index.hbs';
import cast from "url:./img/**/cast/*.png";
import training from "url:./img/**/training/*.png";


Handlebars.registerHelper('every6', function (value) {
    return (value % 6) == 0;
  });

Handlebars.registerHelper('every6offset', function (value) {
    return ((value + 1) % 6) == 0;
  });

var seasons = ['us1', 'us2'];
console.log(cast)

document.getElementById('output').innerHTML = template({
    cast: cast,
    seasons: seasons
});
