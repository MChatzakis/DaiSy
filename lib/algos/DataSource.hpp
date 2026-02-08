#ifndef DATASOURCE_HPP
#define DATASOURCE_HPP

#include "../isax/iSAXTypes.hpp"
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstring>

namespace daisy
{
    // Forward declaration of idx_t type (also defined in SimilaritySearchAlgorithm.hpp)
    using idx_t = unsigned long long;
    /**
     * @brief Abstract interface for data sources (Adapter Pattern)
     *
     * This interface allows algorithms to work with both in-memory and file-based data sources
     * without knowing the underlying implementation details.
     */
    class DataSource
    {
    public:
        virtual ~DataSource() = default;

        /**
         * @brief Get the next single time series record
         *
         * @param record_data Output buffer (must be pre-allocated with size dim)
         * @return true if a record was read, false if no more records available
         */
        virtual bool nextRecord(float *record_data) = 0;

        /**
         * @brief Check if there are more records available
         *
         * @return true if more records are available, false otherwise
         */
        virtual bool hasNext() const = 0;

        /**
         * @brief Get the dimensionality of each time series
         *
         * @return Dimension of time series
         */
        virtual idx_t getDim() const = 0;

        /**
         * @brief Get the total number of time series (if known)
         *
         * @return Total number of time series, or 0 if unknown
         */
        virtual idx_t getTotalRecords() const = 0;

        /**
         * @brief Reset the data source to the beginning
         */
        virtual void reset() = 0;

        /**
         * @brief Get the current position (record index)
         *
         * @return Current position in the data source
         */
        virtual idx_t getCurrentPosition() const = 0;
    };

    /**
     * @brief In-memory data source implementation
     *
     * Wraps an existing in-memory database array for algorithms that expect
     * data to be already loaded in RAM (e.g., Messi).
     */
    class InMemoryDataSource : public DataSource
    {
    private:
        const float *database;
        idx_t n_database;
        idx_t dim;
        idx_t current_pos;

    public:
        /**
         * @brief Construct an in-memory data source
         *
         * @param database Pointer to the database array (row-major: n_database * dim floats)
         * @param n_database Number of time series in the database
         * @param dim Dimensionality of each time series
         */
        InMemoryDataSource(const float *database, idx_t n_database, idx_t dim)
            : database(database), n_database(n_database), dim(dim), current_pos(0)
        {
        }

        bool nextRecord(float *record_data) override
        {
            if (current_pos >= n_database)
                return false;

            std::copy(database + current_pos * dim,
                      database + current_pos * dim + dim,
                      record_data);

            current_pos++;
            return true;
        }

        bool hasNext() const override
        {
            return current_pos < n_database;
        }

        idx_t getDim() const override
        {
            return dim;
        }

        idx_t getTotalRecords() const override
        {
            return n_database;
        }

        void reset() override
        {
            current_pos = 0;
        }

        idx_t getCurrentPosition() const override
        {
            return current_pos;
        }
    };

    /**
     * @brief File-based data source implementation
     *
     * Reads time series data from a binary file record by record.
     * File format: binary float32 array in row-major order (n_database * dim floats).
     */
    class FileDataSource : public DataSource
    {
    private:
        const char *filename;
        idx_t n_database; // Total number of records (0 if unknown, will be determined from file size)
        idx_t dim;
        idx_t current_pos;
        FILE *file;
        bool file_owned; // Whether we opened the file or it was passed to us

        /**
         * @brief Determine total number of records from file size
         */
        void determineTotalRecords()
        {
            if (file == nullptr || dim == 0)
                return;

            long current_pos = ftell(file);
            fseek(file, 0, SEEK_END);
            long file_size = ftell(file);
            fseek(file, current_pos, SEEK_SET);

            if (file_size > 0 && dim > 0)
            {
                n_database = file_size / (sizeof(float) * dim);
            }
        }

    public:
        /**
         * @brief Construct a file data source
         *
         * @param filename Path to the binary file
         * @param dim Dimensionality of each time series
         * @param n_database Total number of time series (0 to auto-detect from file size)
         */
        FileDataSource(const char *filename, idx_t dim, idx_t n_database = 0)
            : filename(filename), n_database(n_database), dim(dim), current_pos(0),
              file(nullptr), file_owned(true)
        {
            file = fopen(filename, "rb");
            if (file == nullptr)
            {
                fprintf(stderr, "Error: Could not open file %s\n", filename);
                throw std::runtime_error("Failed to open file: " + std::string(filename));
            }

            // If n_database not provided, determine from file size
            if (n_database == 0)
            {
                determineTotalRecords();
            }
        }

        /**
         * @brief Construct from an already opened file handle
         *
         * @param file_opened Already opened FILE* handle
         * @param dim Dimensionality of each time series
         * @param n_database Total number of time series (0 to auto-detect from file size)
         * @param file_owned Whether we should close the file in destructor
         */
        FileDataSource(FILE *file_opened, idx_t dim, idx_t n_database = 0, bool file_owned = false)
            : filename(nullptr), n_database(n_database), dim(dim), current_pos(0),
              file(file_opened), file_owned(file_owned)
        {
            if (file == nullptr)
            {
                throw std::runtime_error("FileDataSource: file handle is null");
            }

            // If n_database not provided, determine from file size
            if (n_database == 0)
            {
                determineTotalRecords();
            }
        }

        ~FileDataSource()
        {
            if (file_owned && file != nullptr)
            {
                fclose(file);
                file = nullptr;
            }
        }

        // Delete copy constructor and assignment (file handles shouldn't be copied)
        FileDataSource(const FileDataSource &) = delete;
        FileDataSource &operator=(const FileDataSource &) = delete;

        // Allow move constructor
        FileDataSource(FileDataSource &&other) noexcept
            : filename(other.filename), n_database(other.n_database), dim(other.dim),
              current_pos(other.current_pos), file(other.file), file_owned(other.file_owned)
        {
            other.file = nullptr;
            other.file_owned = false;
        }

        bool nextRecord(float *record_data) override
        {
            if (file == nullptr || !hasNext())
                return false;

            size_t items_read = fread(record_data, sizeof(float), dim, file);
            if (items_read != dim)
            {
                return false; // End of file or error
            }

            current_pos++;
            return true;
        }

        bool hasNext() const override
        {
            if (file == nullptr)
                return false;

            if (n_database > 0)
            {
                return current_pos < n_database;
            }
            else
            {
                // If we don't know total, check if we're at EOF
                // Use a non-const cast since ftell/fseek don't actually modify the file
                FILE *mutable_file = const_cast<FILE *>(file);
                long current_pos_saved = ftell(mutable_file);
                if (current_pos_saved < 0)
                    return false;

                // Try to read one float to see if we're at EOF
                float dummy;
                size_t items_read = fread(&dummy, sizeof(float), 1, mutable_file);
                if (items_read == 0)
                {
                    return false; // EOF
                }

                // Restore position
                fseek(mutable_file, current_pos_saved, SEEK_SET);
                return true;
            }
        }

        idx_t getDim() const override
        {
            return dim;
        }

        idx_t getTotalRecords() const override
        {
            return n_database;
        }

        void reset() override
        {
            if (file != nullptr)
            {
                rewind(file);
                current_pos = 0;
            }
        }

        idx_t getCurrentPosition() const override
        {
            return current_pos;
        }

        /**
         * @brief Get the underlying file handle (for advanced use cases)
         *
         * @return FILE* pointer (nullptr if file is closed)
         */
        FILE *getFileHandle() const
        {
            return file;
        }

        /**
         * @brief Get the filename (if available)
         *
         * @return const char* to filename, or nullptr if not available
         */
        const char *getFilename() const
        {
            return filename;
        }
    };

} // namespace diNoLib

#endif // DATASOURCE_HPP
