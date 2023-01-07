const Handlebars = require("handlebars");
const Bootstrap = require("bootstrap");
var Airtable = require('airtable');
Airtable.configure({
    endpointUrl: 'https://api.airtable.com',
    apiKey: process.env.AIRTABLE_API_KEY
});
var base = Airtable.base(process.env.AIRTABLE_BASE_ID);

import template from './templates/index.hbs';
import cast from "url:./img/**/cast/*.png";
import other_imgs from "url:./img/*.png";
import training from "url:./img/**/training/*.png";
console.log("training: ",training)

Handlebars.registerHelper('every4', function (value) {
    return (value % 4) == 0;
  });

Handlebars.registerHelper('every4offset', function (value) {
    return ((value + 1) % 4) == 0;
  });

var seasons = ['us1', 'us2'];

function getAirData() {
  var ids = []

  return new Promise((resolve, reject) => {
    base(process.env.AIRTABLE_TABLE_NAME).select({
        fields: ["id"],
        view: "Grid view"
    }).eachPage(function page(records, fetchNextPage) {
        // This function (`page`) will get called for each page of records.
        ids.push(records.map(function(record) {
          return record.get('id');
        }));

        // To fetch the next page of records, call `fetchNextPage`.
        // If there are more records, `page` will get called again.
        // If there are no more records, `done` will get called.
        fetchNextPage();

    }, function done(err) {
        if (err) { console.error(err); return; }

        resolve(ids.flat());
    });
  })
}

async function buildBody() {
  const completed_ids = await getAirData();
  // console.log(completed_ids)

  var training_imgs = seasons.map((season) => {
    var trn_ssn = training[season];
    var img_keys = Object.keys(trn_ssn);
    console.log(img_keys)
    // console.log(completed_ids.includes("test"))
    var curr_key = img_keys.filter((img_key) => {
      // console.log(img_key);
      // console.log(!(completed_ids.includes(img_key)));
      return !(completed_ids.includes(img_key));
    })[0];

    return {
      key: curr_key,
      src: trn_ssn[curr_key]
    };
  });

  console.log("Cast: ", cast)
  console.log("Seasons: ", seasons)
  console.log("Training Imgs: ", training_imgs)
  console.log("Completed: ", completed_ids)
  console.log("Other imgs: ", other_imgs)

  document.getElementById('output').innerHTML = template({
    cast: cast,
    seasons: seasons,
    training_imgs: training_imgs,
    other_imgs: other_imgs
  });
}

buildBody()
