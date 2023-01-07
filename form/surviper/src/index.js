addEventListener('fetch', event => {
	event.respondWith(handleRequest(event.request))
})

const FORM_URL = "https://faces.mavec.me"

async function handleRequest(request) {
  const url = new URL(request.url)
  if (url.pathname === "/submit") {
	return submitHandler(request)
  }

  return Response.redirect(FORM_URL)
}

const submitHandler = async request => {
	if (request.method !== "POST") {
	  return new Response("Method Not Allowed", {
		status: 405
	  })
	}

	const body = await request.formData();

	const {
	  id,
	  survivor,
	  season,
	  user
	} = Object.fromEntries(body)

	if (survivor == null) {
		// Don't submit data if there is no survivor specified
		return Response.redirect(FORM_URL)
	}

	// The keys in "fields" are case-sensitive, and
	// should exactly match the field names you set up
	// in your Airtable table, such as "First Name".
	const reqBody = {
	  fields: {
		"id": id,
		"survivor": survivor,
		"season": season,
		"user": user
	  }
	}

	await createAirtableRecord(reqBody)
	return Response.redirect(FORM_URL)
  }

  const createAirtableRecord = body => {
	return fetch(`https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${encodeURIComponent(AIRTABLE_TABLE_NAME)}`, {
	  method: 'POST',
	  body: JSON.stringify(body),
	  headers: {
		Authorization: `Bearer ${AIRTABLE_API_KEY}`,
		'Content-type': `application/json`
	  }
	})
  }
