function SubForm () {
  // appendValues('1IaU8E938msrbuMN38qyjLyKuyRMMI1a3srSFtgLN96s', "Form!A1:D1")
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

const handleSubmit = (event) => {
  event.preventDefault();

  const myForm = event.target;
  const formData = new FormData(myForm);
  
  fetch("/", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(formData).toString(),
  })
    .then(() => console.log("Form successfully submitted"))
    .catch((error) => alert(error));
};

document
  .querySelector("form")
  .addEventListener("submit", handleSubmit);