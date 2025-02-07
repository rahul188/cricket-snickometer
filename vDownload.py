from pytube import YouTube

def download_youtube_video(url, output_path='.'):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path)
        print(f"Downloaded: {yt.title}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=nemrltgpc_A&t=9719s"
    download_youtube_video(video_url, "/path/to/download/directory")