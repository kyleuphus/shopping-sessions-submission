//! Parquet output for the two result tables.
//!
//! The checker (see `starter-*/scripts/run_fixtures.py`) is lenient about integer
//! width and string encoding but strict about two things: the column set must match
//! exactly with no extras, and `day` must be a real Parquet `DATE`. `Date32` is
//! days-since-epoch, which is exactly that.
//!
//! Row order and the number of output files do not matter.
//!
//! `FAST_PQ` selects the encoding once per process. Unset or unknown values
//! use `dict-delta-snappy`. Container exit waits for output write-back, and
//! the smaller Snappy output reduced measured wall time to 0.920x `plain`.
//! `plain` selects the original uncompressed output. `dict` enables dictionaries on
//! low-cardinality columns. `dict-delta` adds delta encoding to integer
//! measures and user IDs, and its `-lz4` and `-snappy` variants compress pages.
//! Every variant skips statistics and uses Int32 for counts only when all
//! counts in the file fit. `revenue_cents` stays Int64.

use std::borrow::Cow;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};

/// One row of `/output/user_daily/`.
pub struct UserDailyRow<'a> {
    pub user_id: Cow<'a, str>,
    /// Days since the Unix epoch (Parquet DATE).
    pub day: i32,
    pub sessions: i64,
    pub events: i64,
    pub revenue_cents: i64,
    /// `None` renders as null, which is required when a person made no purchases.
    /// Null and empty string are distinct to the checker.
    pub top_category: Option<Cow<'a, str>>,
}

/// One row of `/output/campaign_daily/`.
pub struct CampaignDailyRow<'a> {
    pub campaign_id: Cow<'a, str>,
    pub channel: Cow<'a, str>,
    pub day: i32,
    pub sessions: i64,
    pub purchasing_sessions: i64,
    pub revenue_cents: i64,
    pub assisted_sessions: i64,
}

/// `counts` is the type of the count columns: `Int32` or `Int64` (see `count_type`).
pub fn user_daily_schema(counts: DataType) -> Schema {
    Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", counts.clone(), false),
        Field::new("events", counts, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ])
}

/// `counts` is the type of the count columns: `Int32` or `Int64` (see `count_type`).
pub fn campaign_daily_schema(counts: DataType) -> Schema {
    Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", counts.clone(), false),
        Field::new("purchasing_sessions", counts.clone(), false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", counts, false),
    ])
}

/// Int32 if every count value in this file fits in an i32, else Int64. Counts are
/// far below 2^31 at any realistic scale, but a wrong answer scores nothing, so we
/// check rather than assume.
fn count_type(mut values: impl Iterator<Item = i64>) -> DataType {
    if values.all(|v| i32::try_from(v).is_ok()) {
        DataType::Int32
    } else {
        DataType::Int64
    }
}

/// Build a count column of type `ty` (as chosen by `count_type` over the same values).
fn count_column(ty: &DataType, values: impl Iterator<Item = i64>) -> ArrayRef {
    match ty {
        // `count_type` verified every value fits; `try_from` + expect keeps this
        // honest anyway (a panic beats a silently wrapped count).
        DataType::Int32 => Arc::new(Int32Array::from_iter_values(
            values.map(|v| i32::try_from(v).expect("count checked to fit in i32")),
        )),
        _ => Arc::new(Int64Array::from_iter_values(values)),
    }
}

fn write_batch(
    dir: &Path,
    part: usize,
    schema: Schema,
    columns: Vec<ArrayRef>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let schema = Arc::new(schema);
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = File::create(dir.join(format!("part-{part}.parquet")))?;
    static VARIANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let variant = VARIANT.get_or_init(|| std::env::var("FAST_PQ").ok());
    let props = writer_properties(variant.as_deref());
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn writer_properties(variant: Option<&str>) -> WriterProperties {
    use parquet::basic::Encoding;
    use parquet::schema::types::ColumnPath;

    let variant = match variant {
        Some(name @ ("plain" | "dict" | "dict-delta" | "dict-delta-lz4")) => name,
        _ => "dict-delta-snappy",
    };
    let compression = match variant {
        "dict-delta-lz4" => Compression::LZ4_RAW,
        "dict-delta-snappy" => Compression::SNAPPY,
        _ => Compression::UNCOMPRESSED,
    };
    let mut props = WriterProperties::builder()
        .set_compression(compression)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_dictionary_enabled(false);
    let delta = matches!(variant, "dict-delta" | "dict-delta-lz4" | "dict-delta-snappy");
    // Column paths absent from a table's schema are unused by its writer.
    if variant == "dict" || delta {
        for name in ["day", "top_category", "campaign_id", "channel"] {
            props = props.set_column_dictionary_enabled(ColumnPath::from(name), true);
        }
    }
    // These paths inherit dictionary=false, so their explicit encodings apply.
    if delta {
        for name in [
            "sessions",
            "events",
            "revenue_cents",
            "purchasing_sessions",
            "assisted_sessions",
        ] {
            props = props.set_column_encoding(ColumnPath::from(name), Encoding::DELTA_BINARY_PACKED);
        }
        props = props
            .set_column_encoding(ColumnPath::from("user_id"), Encoding::DELTA_LENGTH_BYTE_ARRAY);
    }
    props.build()
}

/// Write one user_daily part file, `part-{part}.parquet`, from any iterator of
/// rows (the pipeline passes several shards' row vectors chained together).
pub fn write_user_daily<'r, 'a: 'r>(
    dir: &Path,
    part: usize,
    rows: impl Iterator<Item = &'r UserDailyRow<'a>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = rows.collect::<Vec<_>>();
    let counts = count_type(rows.iter().flat_map(|r| [r.sessions, r.events]));
    // `from_iter_values` builds a non-nullable array; `from_iter` takes Options and
    // is used only for `top_category`, where null is meaningful and distinct from "".
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.user_id.as_ref()),
        )),
        Arc::new(Date32Array::from_iter_values(rows.iter().map(|r| r.day))),
        count_column(&counts, rows.iter().map(|r| r.sessions)),
        count_column(&counts, rows.iter().map(|r| r.events)),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.revenue_cents),
        )),
        Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.top_category.as_deref()),
        )),
    ];
    write_batch(dir, part, user_daily_schema(counts), columns)
}

pub fn write_campaign_daily(
    dir: &Path,
    rows: &[CampaignDailyRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let counts = count_type(
        rows.iter()
            .flat_map(|r| [r.sessions, r.purchasing_sessions, r.assisted_sessions]),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.campaign_id.as_ref()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.channel.as_ref()),
        )),
        Arc::new(Date32Array::from_iter_values(rows.iter().map(|r| r.day))),
        count_column(&counts, rows.iter().map(|r| r.sessions)),
        count_column(&counts, rows.iter().map(|r| r.purchasing_sessions)),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.revenue_cents),
        )),
        count_column(&counts, rows.iter().map(|r| r.assisted_sessions)),
    ];
    write_batch(dir, 0, campaign_daily_schema(counts), columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    #[test]
    fn parquet_variants_round_trip_both_tables_and_count_widths() {
        let path = std::env::temp_dir().join(format!("fast-variants-{}.parquet", std::process::id()));
        for count in [7, i64::from(i32::MAX) + 1] {
            let counts = count_type([1, count, 3].into_iter());
            let measures = count_column(&counts, [1, count, 3].into_iter());
            let days = Arc::new(Date32Array::from(vec![20000, 20001, 20000]));
            let revenue = Arc::new(Int64Array::from(vec![0, 5_000_000_000, 10]));
            let users = RecordBatch::try_new(
                Arc::new(user_daily_schema(counts.clone())),
                vec![
                    Arc::new(StringArray::from(vec!["repeat", "repeat", "unique"])),
                    days.clone(),
                    measures.clone(),
                    measures.clone(),
                    revenue.clone(),
                    Arc::new(StringArray::from(vec![None, Some(""), Some("books")])),
                ],
            )
            .unwrap();
            let campaigns = RecordBatch::try_new(
                Arc::new(campaign_daily_schema(counts)),
                vec![
                    Arc::new(StringArray::from(vec!["c1", "c1", "c2"])),
                    Arc::new(StringArray::from(vec!["search", "search", "email"])),
                    days,
                    measures.clone(),
                    measures.clone(),
                    revenue,
                    measures,
                ],
            )
            .unwrap();
            for batch in [users, campaigns] {
                for variant in [
                    None,
                    Some("unknown"),
                    Some("plain"),
                    Some("dict"),
                    Some("dict-delta"),
                    Some("dict-delta-lz4"),
                    Some("dict-delta-snappy"),
                ] {
                    let props = writer_properties(variant);
                    if matches!(variant, None | Some("unknown")) {
                        assert_eq!(
                            props.compression(&parquet::schema::types::ColumnPath::from("day")),
                            Compression::SNAPPY
                        );
                    }
                    let mut writer = ArrowWriter::try_new(
                        File::create(&path).unwrap(),
                        batch.schema(),
                        Some(props),
                    )
                    .unwrap();
                    writer.write(&batch).unwrap();
                    writer.close().unwrap();
                    assert_eq!(read_back(&path), batch, "{variant:?}");
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn writes_empty_tables_with_the_right_schema() {
        let dir = std::env::temp_dir().join(format!("fast-out-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_user_daily(&dir.join("user_daily"), 0, [].iter()).unwrap();
        write_campaign_daily(&dir.join("campaign_daily"), &[]).unwrap();
        assert!(dir.join("user_daily/part-0.parquet").is_file());
        assert!(dir.join("campaign_daily/part-0.parquet").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn null_top_category_is_distinct_from_empty_string() {
        // The checker treats null and "" as different values, so this must round-trip.
        let dir = std::env::temp_dir().join(format!("fast-null-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rows = vec![
            UserDailyRow {
                user_id: "u1".into(),
                day: 20000,
                sessions: 1,
                events: 1,
                revenue_cents: 0,
                top_category: None,
            },
            UserDailyRow {
                user_id: "u2".into(),
                day: 20000,
                sessions: 1,
                events: 1,
                revenue_cents: 0,
                top_category: Some(String::new().into()),
            },
        ];
        write_user_daily(&dir.join("user_daily"), 0, rows.iter()).unwrap();
        let b = read_back(&dir.join("user_daily/part-0.parquet"));
        let tc = b.column_by_name("top_category").unwrap();
        let tc = tc.as_any().downcast_ref::<StringArray>().unwrap();
        assert!(tc.is_null(0));
        assert!(!tc.is_null(1));
        assert_eq!(tc.value(1), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Read back a written part file as its single record batch.
    fn read_back(path: &Path) -> RecordBatch {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        batches.into_iter().next().unwrap()
    }

    fn user_row(sessions: i64, events: i64) -> UserDailyRow<'static> {
        UserDailyRow {
            user_id: "u".into(),
            day: 20000,
            sessions,
            events,
            revenue_cents: 5_000_000_000, // > i32::MAX: must stay Int64
            top_category: None,
        }
    }

    fn dtype(b: &RecordBatch, name: &str) -> DataType {
        b.schema().field_with_name(name).unwrap().data_type().clone()
    }

    #[test]
    fn counts_are_int32_when_they_fit_and_int64_otherwise() {
        let dir = std::env::temp_dir().join(format!("fast-width-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Fits: counts narrow to Int32, revenue stays Int64, day is Date32.
        let rows = vec![user_row(1, i32::MAX as i64)];
        write_user_daily(&dir.join("a"), 0, rows.iter()).unwrap();
        let b = read_back(&dir.join("a/part-0.parquet"));
        assert_eq!(dtype(&b, "day"), DataType::Date32);
        assert_eq!(dtype(&b, "sessions"), DataType::Int32);
        assert_eq!(dtype(&b, "events"), DataType::Int32);
        assert_eq!(dtype(&b, "revenue_cents"), DataType::Int64);
        let ev = b.column_by_name("events").unwrap();
        assert_eq!(ev.as_any().downcast_ref::<Int32Array>().unwrap().value(0), i32::MAX);
        let rev = b.column_by_name("revenue_cents").unwrap();
        assert_eq!(
            rev.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
            5_000_000_000
        );

        // One value overflows i32: the whole file falls back to Int64, value intact.
        let rows = vec![user_row(1, 2), user_row(3, i32::MAX as i64 + 1)];
        write_user_daily(&dir.join("b"), 0, rows.iter()).unwrap();
        let b = read_back(&dir.join("b/part-0.parquet"));
        assert_eq!(dtype(&b, "sessions"), DataType::Int64);
        let ev = b.column_by_name("events").unwrap();
        let ev = ev.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ev.value(1), i32::MAX as i64 + 1);

        // Campaign table: small counts narrow; any overflow falls back.
        let mk = |assisted| CampaignDailyRow {
            campaign_id: "c".into(),
            channel: "x".into(),
            day: 20000,
            sessions: 1,
            purchasing_sessions: 1,
            revenue_cents: 1,
            assisted_sessions: assisted,
        };
        write_campaign_daily(&dir.join("c"), &[mk(7)]).unwrap();
        let b = read_back(&dir.join("c/part-0.parquet"));
        assert_eq!(dtype(&b, "assisted_sessions"), DataType::Int32);
        assert_eq!(dtype(&b, "revenue_cents"), DataType::Int64);
        write_campaign_daily(&dir.join("d"), &[mk(7), mk(i64::MAX)]).unwrap();
        let b = read_back(&dir.join("d/part-0.parquet"));
        assert_eq!(dtype(&b, "assisted_sessions"), DataType::Int64);
        let a = b.column_by_name("assisted_sessions").unwrap();
        assert_eq!(a.as_any().downcast_ref::<Int64Array>().unwrap().value(1), i64::MAX);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
