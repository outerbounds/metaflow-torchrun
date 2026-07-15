import codecs
from threading import Thread


def _drain_stream(source, destination, errors):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    destination_failed = False

    try:
        while True:
            chunk = source.read1()
            if not chunk:
                break

            if destination_failed:
                # Keep draining the child even if forwarding failed so that the
                # child cannot remain blocked on a full pipe.
                continue

            try:
                text = decoder.decode(chunk)
                if text:
                    destination.write(text)
                destination.flush()
            except BaseException as error:
                errors.append(error)
                destination_failed = True

        if not destination_failed:
            text = decoder.decode(b"", final=True)
            if text:
                destination.write(text)
                destination.flush()
    except BaseException as error:
        errors.append(error)
        try:
            source.close()
        except Exception:
            pass


def stream_subprocess_output(process, stdout_destination, stderr_destination):
    """Drain and forward both subprocess output streams concurrently."""
    errors = []
    threads = (
        Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_destination, errors),
            name="torchrun-stdout",
        ),
        Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_destination, errors),
            name="torchrun-stderr",
        ),
    )

    for thread in threads:
        thread.start()

    returncode = process.wait()

    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    return returncode
