import httpclient, json, base64

proc webEncodeData(data, mimeType: string): string =
  return "data:image/" & mimeType & ";base64," & base64.encode(data, newLine="")

proc postJson(url: string, params: JsonNode) =
  let body = $params
  let client = newHttpClient()
  client.headers = newHttpHeaders({
    "content-type": "application/json",
    "content-length": $body.len
  })
  let response = client.request(url, httpMethod = HttpPost, body = body)
  if response.code != Http200:
    raise newException(IOError, "Failed to post json data")

type
  VisdomClient = object
    host: string
    port: int

proc newVisdomClient*(host: string = "localhost", port: int = 8097): VisdomClient =
  ## Prepare a visdom client for visualization
  result.host = host
  result.port = port

proc sendEvent(self: VisdomClient, opts: JsonNode, data: JsonNode, window: string) =
  let params = %*{
    "eid": "main",
    "opts": opts,
    "data": data
  }

  if window.len > 0:
    params["win"] = % window

  let url = "http://" & self.host & ":" & $self.port & "/events"
  postJson(url, params)

proc image*(vis: VisdomClient,
  img: Tensor[uint8],
  window: string = "",
  caption: string = "",
  title: string = "") =
  ## Show image into visdom with the given title and specified window

  let opts = %*{
    "title": if title.len > 0: title else: window,
    "height": img.height,
    "width": img.width
  }

  let data = %*[{
    "content": {
      "src": img.toJPG().webEncodeData("image/jpg"),
      "caption": caption
    },
    "type": "image"
  }]

  vis.sendEvent(opts, data, window)
