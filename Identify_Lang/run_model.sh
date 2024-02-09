if [ "$1" = "train" ]; then
  if [ $# -lt 4 ]; then
    python "$1".py "$2" "$3"
  else
    python "$1".py "$2" "$3" "$4"
  fi
else
  python "$1".py "$2" "$3" "$4"
fi
