function appendValues(spreadsheetId, range, valueInputOption, _values, callback) {
  let values = [
    [
      // Cell values ...
    ],
    // Additional rows ...
  ];
  values = _values;
  const body = {
    values: values,
  };
  try {
    gapi.client.sheets.spreadsheets.values.append({
      spreadsheetId: spreadsheetId,
      range: range,
      valueInputOption: valueInputOption,
      resource: body,
    }).then((response) => {
      const result = response.result;
      console.log(`${result.updates.updatedCells} cells appended.`);
      if (callback) callback(response);
    });
  } catch (err) {
    document.getElementById('content').innerText = err.message;
    return;
  }
}

function SubForm (){
  appendValues('1IaU8E938msrbuMN38qyjLyKuyRMMI1a3srSFtgLN96s', "Form!A1:D1")
    // $.ajax({
    //     url:'https://api.apispreadsheets.com/data/410/',
    //     type:'post',
    //     data:$("#myForm").serializeArray(),
    //     success: function(){
    //       alert("Form Data Submitted :)")
    //     },
    //     error: function(){
    //       alert("There was an error :(")
    //     }
    // });
}
