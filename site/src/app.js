const Handlebars = require("handlebars");
const Bootstrap = require("bootstrap");
var _ = require('lodash');
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

  var training_imgs = _.fromPairs(seasons.map((season) => {
    var trn_ssn = training[season];
    var img_keys = Object.keys(trn_ssn);
    console.log(img_keys)
    // console.log(completed_ids.includes("test"))
    var curr_key = img_keys.filter((img_key) => {
      // console.log(img_key);
      // console.log(!(completed_ids.includes(img_key)));
      return !(completed_ids.includes(img_key));
    })[0];

    return [season, {
      key: curr_key,
      src: trn_ssn[curr_key]
    }];
  }));

  console.log("Training Imgs: ", training_imgs)
  console.log("Completed: ", completed_ids)
  console.log("Other imgs: ", other_imgs)

  seasons = seasons.filter((val, idx) => {
    return training_imgs[val]['key'] != undefined
  })
  console.log("Seasons: ", seasons);
  console.log("Cast: ", cast);
  var f_cast = _.pick(cast, seasons);
  var f_training_imgs = _.pick(training_imgs, seasons);

  // Calculate proportion of images already labelled
  var label_counts = _.fromPairs(seasons.map((season) => {
    var count = 0;
    var completed = 0;
    Object.keys(training[season]).forEach((key) => {
      count += 1;
      if(completed_ids.includes(key)) completed += 1;
    })

    return [season, {
      valuenow: completed,
      valuemin: 0,
      valuemax: count,
      width: _.round(100 * completed / count, 0) + "%"
    }];
  }));
  console.log(label_counts);
  
  document.getElementById('output').innerHTML = template({
    seasons: seasons,
    cast: f_cast,
    training_imgs: f_training_imgs,
    other_imgs: other_imgs,
    label_counts: label_counts
  });
}

buildBody()
